import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.api.config import Config
from src.api.content_store import save_extracted_pickle
from src.api.dataHandler import (
    extract_document_info,
    decrypt_data,
    fileProcessor,
    get_azure_docs,
    get_s3_client,
    get_subscription_pii_setting,
    mask_document_content,
    read_s3_file,
    resolve_subscription_id,
    update_extraction_metadata,
    update_pii_stats,
)
from src.core.db.mongo_provider import MongoProviderError, mongo_provider
from src.api.document_status import init_document_record, set_error, update_document_fields, update_stage
from src.api.pipeline_models import ExtractedDocument
from src.api.statuses import (
    STATUS_DELETED,
    STATUS_EXTRACTION_COMPLETED,
    STATUS_EXTRACTION_FAILED,
    STATUS_UNDER_REVIEW,
)
from src.storage.azure_blob_client import BlobDownloadError, CredentialError, normalize_blob_name

logger = logging.getLogger(__name__)


def _resolve_mongo_db(mongo_db: Optional[Any] = None) -> Any:
    if mongo_db is not None:
        return mongo_db
    return mongo_provider.get_db()


def _log_db_diagnostics(db_handle: Any) -> None:
    logger.info(
        "Extraction DB handle: type=%s id=%s module=%s has_list_collection_names=%s has_getitem=%s",
        type(db_handle).__name__,
        id(db_handle),
        type(db_handle).__module__,
        hasattr(db_handle, "list_collection_names"),
        hasattr(db_handle, "__getitem__"),
    )


def _build_extraction_summary(extracted_obj: Any) -> Dict[str, Any]:
    total_chars = 0
    total_pages = 0
    total_chunks = 0

    def handle_extracted(value: Any) -> None:
        nonlocal total_chars, total_pages, total_chunks
        if isinstance(value, ExtractedDocument):
            text_val = value.full_text or ""
            if not text_val:
                text_val = "\n".join([sec.text for sec in value.sections if sec.text])
            total_chars += len(text_val)
            total_chunks += len(value.sections)
            for sec in value.sections:
                total_pages = max(total_pages, sec.end_page or sec.start_page or 0)
        elif isinstance(value, str):
            total_chars += len(value)
            total_chunks += 1
        elif isinstance(value, dict):
            inner_text = value.get("text") or value.get("content")
            if isinstance(inner_text, str):
                total_chars += len(inner_text)
                total_chunks += 1
            elif isinstance(inner_text, list):
                total_chunks += len(inner_text)
        elif isinstance(value, list):
            total_chunks += len(value)

    if isinstance(extracted_obj, dict):
        for entry in extracted_obj.values():
            handle_extracted(entry)
    else:
        handle_extracted(extracted_obj)

    return {
        "pages": total_pages or None,
        "chunks": total_chunks or None,
        "chars": total_chars,
        "language": None,
    }


def _set_document_status(
    document_id: str,
    status: str,
    error_msg: Optional[str] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    fields: Dict[str, Any] = {"status": status, "updated_at": time.time()}
    if error_msg:
        fields["training_error"] = error_msg
        fields["training_failed_at"] = time.time()
    else:
        fields["training_error"] = None
    if extra_fields:
        fields.update(extra_fields)
    update_document_fields(document_id, fields)
    logger.info("Document %s status updated to %s", document_id, status)


def _extract_from_connector(doc_id: str, doc_data: Dict[str, Any], conn_data: Dict[str, Any]) -> Dict[str, Any]:
    profile_id = str(doc_data.get("profile")) if doc_data.get("profile") else None
    subscription_candidate = (
        doc_data.get("subscriptionId")
        or doc_data.get("subscription_id")
        or doc_data.get("subscription")
        or (conn_data.get("subscriptionId") if isinstance(conn_data, dict) else None)
        or (conn_data.get("subscription") if isinstance(conn_data, dict) else None)
    )

    try:
        subscription_id = resolve_subscription_id(doc_id, subscription_candidate)
    except Exception as exc:  # noqa: BLE001
        _set_document_status(doc_id, STATUS_EXTRACTION_FAILED, str(exc))
        return {"document_id": doc_id, "status": STATUS_EXTRACTION_FAILED, "error": str(exc)}

    pii_masking_enabled = get_subscription_pii_setting(subscription_id)
    logger.info("Document %s (subscription %s): PII masking=%s", doc_id, subscription_id, pii_masking_enabled)

    all_extracted_docs: Dict[str, Any] = {}
    try:
        if doc_data.get("type") == "S3":
            bk_name = conn_data["s3_details"]["bucketName"]
            region = conn_data["s3_details"]["region"]
            ak = decrypt_data(conn_data["s3_details"]["accessKey"]).split("\x0c")[0].strip()
            sk = decrypt_data(conn_data["s3_details"]["secretKey"]).split("\x08")[0].strip()
            s3 = get_s3_client(ak, sk, region)
            if not s3:
                raise ValueError("Failed to create S3 client")

            objs = s3.list_objects_v2(Bucket=bk_name)
            file_keys = [obj["Key"] for obj in objs.get("Contents", []) if obj["Key"] == doc_data.get("name")]
            if not file_keys:
                raise ValueError("File not found in S3")

            doc_content = read_s3_file(s3, bk_name, file_keys[0])
            if doc_content is None:
                raise ValueError("Failed to read S3 file")

            extracted_doc = fileProcessor(doc_content, file_keys[0])
            if not extracted_doc:
                raise ValueError("Content extraction failed")

            all_extracted_docs.update(extracted_doc)
        elif doc_data.get("type") == "LOCAL":
            doc_name = doc_data.get("name", "")
            if not doc_name:
                raise ValueError("No filename specified")

            all_connector_files = conn_data.get("locations", [])
            matching_files: List[str] = []
            for file_path in all_connector_files:
                file_name_only = file_path.split("/")[-1] if "/" in file_path else file_path
                if file_name_only == doc_name:
                    matching_files.append(file_path)
                    break

            if not matching_files:
                raise ValueError(f"Exact file match not found: {doc_name}")
            if len(matching_files) > 1:
                raise ValueError("Multiple file matches")

            file_path = matching_files[0]
            file_key = normalize_blob_name(file_path, container_name=Config.AzureBlob.DOCUMENT_CONTAINER_NAME)
            doc_content = get_azure_docs(file_key, document_id=doc_id)
            if doc_content is None:
                raise ValueError(f"Failed to read file {file_key}")

            extracted_doc = fileProcessor(doc_content, file_path)
            if not extracted_doc:
                raise ValueError("Content extraction failed")

            all_extracted_docs.update(extracted_doc)
        else:
            raise ValueError(f"Unsupported connector type: {doc_data.get('type')}")
    except CredentialError as exc:
        _set_document_status(doc_id, STATUS_EXTRACTION_FAILED, f"CredentialError: {exc}")
        raise
    except BlobDownloadError as exc:
        _set_document_status(doc_id, STATUS_EXTRACTION_FAILED, f"{exc.__class__.__name__}: {exc}")
        return {
            "document_id": doc_id,
            "status": STATUS_EXTRACTION_FAILED,
            "error": f"{exc.__class__.__name__}: {exc}",
        }
    except Exception as exc:  # noqa: BLE001
        _set_document_status(doc_id, STATUS_EXTRACTION_FAILED, str(exc))
        return {"document_id": doc_id, "status": STATUS_EXTRACTION_FAILED, "error": str(exc)}

    if not all_extracted_docs:
        _set_document_status(doc_id, STATUS_EXTRACTION_FAILED, "No content extracted")
        return {"document_id": doc_id, "status": STATUS_EXTRACTION_FAILED, "error": "No content extracted"}

    masked_docs = all_extracted_docs
    pii_count = 0
    pii_items: List[Any] = []
    if pii_masking_enabled:
        masked_docs, pii_count, _high_conf, pii_items = mask_document_content(all_extracted_docs)
        update_pii_stats(doc_id, pii_count, False, pii_items)
    else:
        update_pii_stats(doc_id, 0, False, [])

    if pii_masking_enabled:
        for fname, content in masked_docs.items():
            if isinstance(content, dict) and "texts" in content:
                texts = content.get("texts") or []
                if texts:
                    from src.api.dataHandler import encode_with_fallback

                    content["embeddings"] = encode_with_fallback(
                        texts,
                        convert_to_numpy=True,
                        normalize_embeddings=False,
                    )
                masked_docs[fname] = content

    try:
        save_info = save_extracted_pickle(doc_id, masked_docs)
        update_extraction_metadata(doc_id, subscription_id, save_info.get("path"), save_info.get("sha256"))
    except Exception as exc:  # noqa: BLE001
        _set_document_status(doc_id, STATUS_EXTRACTION_FAILED, f"Failed to persist extracted content: {exc}")
        return {"document_id": doc_id, "status": STATUS_EXTRACTION_FAILED, "error": str(exc)}

    extra_fields = {
        "subscription_id": subscription_id,
        "profile_id": profile_id,
        "extracted_pickle_path": save_info.get("path"),
        "extracted_hash": save_info.get("sha256"),
    }
    _set_document_status(doc_id, STATUS_EXTRACTION_COMPLETED, extra_fields=extra_fields)

    summary = _build_extraction_summary(masked_docs)
    return {
        "document_id": doc_id,
        "status": STATUS_EXTRACTION_COMPLETED,
        "pickle_path": save_info.get("path"),
        "summary": summary,
        "doc_name": doc_data.get("name", "Unknown"),
    }


def extract_documents(mongo_db: Optional[Any] = None) -> Dict[str, Any]:
    db_handle = _resolve_mongo_db(mongo_db)
    _log_db_diagnostics(db_handle)
    try:
        doc_coll = extract_document_info(mongo_db=db_handle)
        if not doc_coll:
            return {"status": "no_documents", "message": "No documents found for extraction"}

        allowed_statuses = {STATUS_UNDER_REVIEW, STATUS_EXTRACTION_FAILED}
        eligible_docs = {
            doc_id: doc_info
            for doc_id, doc_info in doc_coll.items()
            if doc_info.get("dataDict", {}).get("status") in allowed_statuses
        }

        if not eligible_docs:
            return {"status": "no_documents", "message": "No documents eligible for extraction"}

        results = {"successful": [], "failed": [], "total": len(eligible_docs)}
        for doc_id, doc_info in eligible_docs.items():
            if doc_info.get("dataDict", {}).get("status") == STATUS_DELETED:
                continue
            try:
                res = _extract_from_connector(doc_id, doc_info.get("dataDict", {}), doc_info.get("connDict", {}))
            except CredentialError as exc:
                logger.error("Credential error during extraction; failing batch: %s", exc)
                return {"status": "error", "message": f"CredentialError: {exc}", "results": None}
            if res.get("status") == STATUS_EXTRACTION_COMPLETED:
                results["successful"].append(res)
            else:
                results["failed"].append(res)

        return {"status": "completed", "results": results}
    except (MongoProviderError, RuntimeError):
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Extraction process failed: %s", exc, exc_info=True)
        return {"status": "error", "message": str(exc), "results": None}


def extract_single_document(doc_id: str, mongo_db: Optional[Any] = None) -> Dict[str, Any]:
    db_handle = _resolve_mongo_db(mongo_db)
    _log_db_diagnostics(db_handle)
    doc_coll = extract_document_info(mongo_db=db_handle)
    if not doc_coll or doc_id not in doc_coll:
        return {"status": "not_found", "message": f"Document {doc_id} not found"}
    doc_info = doc_coll[doc_id]
    return _extract_from_connector(doc_id, doc_info.get("dataDict", {}), doc_info.get("connDict", {}))


def extract_uploaded_document(
    *,
    document_id: str,
    file_bytes: bytes,
    filename: str,
    subscription_id: Optional[str] = None,
    profile_id: Optional[str] = None,
    profile_name: Optional[str] = None,
    doc_type: Optional[str] = None,
    content_type: Optional[str] = None,
    content_size: Optional[int] = None,
) -> Dict[str, Any]:
    init_document_record(
        document_id=document_id,
        subscription_id=subscription_id,
        profile_id=profile_id,
        doc_type=doc_type,
        filename=filename,
        content_type=content_type,
        size=content_size,
    )
    if profile_name:
        update_document_fields(document_id, {"profile_name": profile_name, "metadata.profile_name": profile_name})

    update_stage(document_id, "extraction", {"status": "IN_PROGRESS", "started_at": time.time(), "error": None})

    try:
        extracted = fileProcessor(file_bytes, filename)
        if not extracted:
            raise ValueError("No content extracted from file")
    except Exception as exc:  # noqa: BLE001
        set_error(document_id, "extraction", exc)
        _set_document_status(document_id, STATUS_EXTRACTION_FAILED, str(exc))
        raise

    try:
        save_info = save_extracted_pickle(document_id, extracted)
        update_extraction_metadata(document_id, subscription_id, save_info.get("path"), save_info.get("sha256"))
    except Exception as exc:  # noqa: BLE001
        set_error(document_id, "extraction", exc)
        _set_document_status(document_id, STATUS_EXTRACTION_FAILED, str(exc))
        raise

    stats = _build_extraction_summary(extracted)
    update_stage(
        document_id,
        "extraction",
        {
            "status": "COMPLETED",
            "completed_at": time.time(),
            "error": None,
            "stats": stats,
        },
    )
    blob_name = None
    if save_info.get("path"):
        blob_name = Path(save_info.get("path")).name
    update_document_fields(
        document_id,
        {
            "blob": {
                "container": "document-content",
                "name": blob_name,
                "etag": None,
                "size": save_info.get("size"),
            },
            "extracted_pickle_path": save_info.get("path"),
            "extracted_hash": save_info.get("sha256"),
        },
    )

    _set_document_status(
        document_id,
        STATUS_EXTRACTION_COMPLETED,
        extra_fields={"extracted_pickle_path": save_info.get("path"), "extracted_hash": save_info.get("sha256")},
    )

    update_stage(
        document_id,
        "screening.security",
        {
            "status": "PENDING",
            "risk_level": None,
            "started_at": None,
            "completed_at": None,
            "error": None,
        },
    )
    update_stage(
        document_id,
        "screening",
        {"overall_status": "PENDING", "last_run_id": None, "updated_at": time.time()},
    )
    update_stage(
        document_id,
        "embedding",
        {"status": "PENDING", "reason": None, "started_at": None, "completed_at": None, "error": None},
    )

    return {
        "document_id": document_id,
        "status": STATUS_EXTRACTION_COMPLETED,
        "pickle_path": save_info.get("path"),
        "summary": stats,
        "extraction": {"status": "COMPLETED", "stats": stats},
        "blob": {"container": "document-content", "name": blob_name},
    }
