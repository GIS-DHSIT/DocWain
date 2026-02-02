
import copy
import csv
import hashlib
import json
import logging
import os
import re
import subprocess
import time
import uuid
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3 as b3
import numpy as np
import pandas as pd
from Crypto.Cipher import AES
from bson.objectid import ObjectId
from pymongo import MongoClient, errors
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from urllib.parse import urlparse
try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

from src.api.config import Config
from src.core.db.mongo_provider import MongoProviderError, mongo_db_proxy, mongo_provider
from src.api.context_understanding import ContextUnderstanding
from src.api.pii_masking import mask_document_content
from src.api.dw_document_extractor import DocumentExtractor
from src.api.content_store import delete_extracted_pickle, save_extracted_pickle
from src.api.pipeline_models import ChunkCandidate, ChunkRecord, ExtractedDocument, Section
from src.api.pipeline_state import is_screening_stage
from src.api.vector_store import QdrantVectorStore, build_collection_name, compute_chunk_id
from src.chunking.section_chunker import SectionChunker, normalize_text
from src.metadata.normalizer import MetadataNormalizationError, normalize_payload_metadata
from src.metrics.ai_metrics import get_metrics_store
from src.metrics.telemetry import METRICS_V2_ENABLED, telemetry_store
from azure.core.exceptions import ResourceNotFoundError

from src.storage.azure_blob_client import (
    classify_blob_error,
    extract_azure_error_details,
    get_azure_blob,
    iter_blob_name_candidates,
    normalize_blob_name,
    sanitize_blob_url,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

mongoClient: Optional[MongoClient] = None
db = mongo_db_proxy

# Lazy-loaded globals to avoid heavy initialization during import
docEx = None
_MODEL = None
_MODEL_DEVICE = None
_QDRANT_CLIENT = None
_VECTOR_STORE = None
_HASH_VECTORIZER = None
logger = logging.getLogger(__name__)

'-------------------------------modified by maha/maria-----------------------'
'---------------------------------new function for checking PII status in mongodb--------------------'


def get_subscription_pii_setting(subscription_id: str) -> bool:
    """
    Fetch PII enabled/disabled setting from subscription in MongoDB.

    Args:
        subscription_id: The subscription ID to check

    Returns:
        bool: True if PII masking is enabled, False if disabled
        Default to True if setting not found (safe default)
    """
    try:
        # Check if subscriptions collection exists in Config
        subscriptions_collection = getattr(Config.MongoDB, 'SUBSCRIPTIONS', 'subscriptions')
        collection = db[subscriptions_collection]

        # Try to find subscription by _id or subscriptionId field
        subscription = None

        # Try ObjectId first if valid
        if ObjectId.is_valid(subscription_id):
            subscription = collection.find_one({"_id": ObjectId(subscription_id)})

        # If not found, try string matching
        if not subscription:
            subscription = collection.find_one({"subscriptionId": subscription_id})

        # If still not found, try _id as string
        if not subscription:
            subscription = collection.find_one({"_id": subscription_id})

        if subscription:
            # Check for pii_enabled field (can be pii_enabled, piiEnabled, or enable_pii)
            pii_enabled = subscription.get('pii_enabled') or subscription.get('piiEnabled') or subscription.get(
                'enable_pii')

            if pii_enabled is not None:
                logging.info(
                    f"Subscription {subscription_id}: PII masking is {'ENABLED' if pii_enabled else 'DISABLED'}")
                return bool(pii_enabled)
            else:
                logging.warning(f"Subscription {subscription_id}: PII setting not found, defaulting to ENABLED")
                return True  # Safe default - enable PII masking if not specified
        else:
            logging.warning(
                "PII masking defaulted to enabled because subscription not found (subscription_id=%s)",
                subscription_id,
            )
            return True  # Safe default

    except Exception as e:
        logging.error(f"Error fetching PII setting for subscription {subscription_id}: {e}")
        return True  # Safe default - enable PII masking on error


def normalize_embedding_matrix(raw_vectors, expected_dim=None):
    """
    Normalize a batch of embeddings into List[List[float]].

    Acceptable inputs:
    - np.ndarray of shape (n, dim)
    - list[list[float]] (or list[tuple])
    - list[np.ndarray]

    Everything else is treated as a bug in the upstream pipeline.
    """
    # np.ndarray -> list of lists
    if isinstance(raw_vectors, np.ndarray):
        raw_vectors = raw_vectors.tolist()

    if not isinstance(raw_vectors, (list, tuple)) or not raw_vectors:
        raise TypeError(f"Embedding batch must be non-empty list/tuple, got {type(raw_vectors)}")

    normalized_vectors = []
    dim = None

    for idx, row in enumerate(raw_vectors):
        # Single row can be np.array, list, or tuple
        if isinstance(row, np.ndarray):
            row = row.tolist()

        if not isinstance(row, (list, tuple)):
            raise TypeError(f"Embedding row at index {idx} must be list/tuple, got {type(row)}")

        vec = [float(x) for x in row]

        if dim is None:
            dim = len(vec)
        elif len(vec) != dim:
            raise ValueError(f"Inconsistent embedding size at index {idx}: {len(vec)} vs {dim}")

        normalized_vectors.append(vec)

    if expected_dim is not None and dim is not None and dim != expected_dim:
        raise ValueError(f"Embedding dimension mismatch: expected {expected_dim}, got {dim}")

    return normalized_vectors, (dim or expected_dim)


def build_sparse_vectors(texts: List[str]) -> List[Dict[str, List[float]]]:
    """Build hashing-based sparse vectors for keyword search."""
    vectorizer = get_hash_vectorizer()
    matrix = vectorizer.transform(texts)
    sparse_vectors = []
    for row in matrix:
        coo = row.tocoo()
        sparse_vectors.append(
            {
                "indices": coo.col.tolist(),
                "values": coo.data.astype(np.float32).tolist(),
            }
        )
    return sparse_vectors


def compute_section_summaries(
    chunks: List[str], chunk_metadata: List[dict], extracted: Optional[ExtractedDocument] = None
) -> List[str]:
    """Generate per-section summaries for chunk payloads."""
    if not chunks or not chunk_metadata:
        return []

    ctx = ContextUnderstanding()

    if extracted:
        doc_summary = ctx.summarize_document(extracted)
        section_summaries = doc_summary.get("section_summaries", {})
        return ContextUnderstanding.attach_summaries_to_chunks(chunk_metadata, section_summaries)

    section_map: Dict[str, Dict[str, Any]] = {}
    for chunk, meta in zip(chunks, chunk_metadata):
        section_id = meta.get("section_id") or meta.get("section_title") or "section"
        record = section_map.setdefault(
            section_id,
            {"title": meta.get("section_title") or "Section", "page": meta.get("page_number") or 0, "texts": []},
        )
        record["texts"].append(chunk)

    sections = [
        Section(
            section_id=sec_id,
            title=rec["title"],
            level=1,
            start_page=rec["page"],
            end_page=rec["page"],
            text="\n".join(rec["texts"]),
        )
        for sec_id, rec in section_map.items()
    ]
    section_summaries = {sec.section_id: ctx.summarize_section(sec) for sec in sections}
    return ContextUnderstanding.attach_summaries_to_chunks(chunk_metadata, section_summaries)



def create_mongo_client() -> MongoClient:
    """
    Create a Mongo client using the shared provider.
    """
    mongo_provider.init()
    return mongo_provider.get_client()


def get_mongo_db():
    return mongo_provider.init()


def _resolve_mongo_db(mongo_db=None):
    if mongo_db is not None:
        return mongo_db
    return mongo_provider.get_db()


def get_doc_extractor():
    """Lazy init for document extractor."""
    global docEx
    if docEx is None:
        docEx = DocumentExtractor()
    return docEx


def _torch_cuda_available() -> bool:
    try:
        return bool(torch) and bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _preferred_embedding_device() -> str:
    env_device = (os.getenv("EMBEDDING_DEVICE") or "").strip().lower()
    if env_device:
        return env_device
    return "cuda" if _torch_cuda_available() else "cpu"


def _is_meta_tensor_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "meta tensor" in msg or "cannot copy out of meta tensor" in msg


def _is_cuda_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "cuda out of memory" in msg or "cuda error: out of memory" in msg


def _resolve_torch_dtype(device: str):
    if not torch:
        return None
    raw = (os.getenv("EMBEDDING_TORCH_DTYPE") or "").strip().lower()
    if raw in {"fp16", "float16", "half"}:
        return torch.float16
    if raw in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if raw in {"fp32", "float32", "full"}:
        return torch.float32
    # Only set dtype when explicitly requested; rely on HF defaults otherwise.
    return None


def _model_kwargs_for_device(device: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    dtype = _resolve_torch_dtype(device)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    return kwargs


def _embedding_candidates() -> List[str]:
    candidates: List[str] = []
    for name in getattr(Config.Model, "SENTENCE_TRANSFORMERS_CANDIDATES", []) or []:
        if name and name not in candidates:
            candidates.append(str(name))
    if not candidates:
        fallback_name = getattr(Config.Model, "EMBEDDING_MODEL", None) or getattr(
            Config.Model, "SENTENCE_TRANSFORMERS", "BAAI/bge-large-en-v1.5"
        )
        candidates.append(str(fallback_name))
    return candidates


def _load_sentence_transformer(name: str, device: str) -> SentenceTransformer:
    logging.info("Loading sentence transformer model: %s (device=%s)", name, device)
    model_kwargs = _model_kwargs_for_device(device)
    try:
        if model_kwargs:
            return SentenceTransformer(name, device=device, model_kwargs=model_kwargs)
        return SentenceTransformer(name, device=device)
    except Exception as exc:  # noqa: BLE001
        if device == "cpu" and model_kwargs and _is_meta_tensor_error(exc):
            logging.warning(
                "Model load on cpu failed (%s); retrying without model kwargs",
                exc,
            )
            return SentenceTransformer(name, device="cpu")
        if device != "cpu" and (_is_meta_tensor_error(exc) or _is_cuda_oom(exc)):
            logging.warning(
                "Model load on %s failed (%s); retrying on cpu for stability",
                device,
                exc,
            )
            cpu_kwargs = _model_kwargs_for_device("cpu")
            try:
                if cpu_kwargs:
                    return SentenceTransformer(name, device="cpu", model_kwargs=cpu_kwargs)
                return SentenceTransformer(name, device="cpu")
            except Exception as cpu_exc:  # noqa: BLE001
                if cpu_kwargs and _is_meta_tensor_error(cpu_exc):
                    logging.warning(
                        "Model load on cpu failed (%s); retrying without model kwargs",
                        cpu_exc,
                    )
                    return SentenceTransformer(name, device="cpu")
                raise
        raise


def get_model(*, reload: bool = False, device: Optional[str] = None):
    """Lazy init for sentence transformer model with robust device fallback."""
    global _MODEL, _MODEL_DEVICE
    candidates = _embedding_candidates()
    target_device = (device or _preferred_embedding_device()).strip().lower()

    should_reload = reload or _MODEL is None
    if _MODEL is not None and _MODEL_DEVICE and target_device and _MODEL_DEVICE != target_device:
        should_reload = True

    if should_reload:
        last_error: Optional[Exception] = None
        for idx, candidate_name in enumerate(candidates):
            try:
                model = _load_sentence_transformer(candidate_name, target_device)
                _MODEL = model
                _MODEL_DEVICE = getattr(model, "_target_device", None) or target_device
                expected_dim = getattr(Config.Model, "EMBEDDING_DIM", None)
                model_dim = _MODEL.get_sentence_embedding_dimension()
                if expected_dim and model_dim != expected_dim:
                    logging.warning(
                        "Configured EMBEDDING_DIM=%s but model dimension is %s",
                        expected_dim,
                        model_dim,
                    )
                logging.info(
                    "Loaded sentence transformer model: %s (dim=%s, device=%s)",
                    candidate_name,
                    model_dim,
                    _MODEL_DEVICE,
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logging.warning(
                    "Failed to load sentence transformer candidate %s on %s: %s",
                    candidate_name,
                    target_device,
                    exc,
                )
                # On meta tensor errors, immediately try CPU for the same candidate.
                if target_device != "cpu" and _is_meta_tensor_error(exc):
                    try:
                        model = _load_sentence_transformer(candidate_name, "cpu")
                        _MODEL = model
                        _MODEL_DEVICE = getattr(model, "_target_device", None) or "cpu"
                        model_dim = _MODEL.get_sentence_embedding_dimension()
                        logging.info(
                            "Loaded sentence transformer model on cpu after meta error: %s (dim=%s)",
                            candidate_name,
                            model_dim,
                        )
                        break
                    except Exception as cpu_exc:  # noqa: BLE001
                        last_error = cpu_exc
                        logging.warning(
                            "CPU retry failed for candidate %s: %s",
                            candidate_name,
                            cpu_exc,
                        )
                        continue
                # Try the next candidate.
                if idx == len(candidates) - 1 and last_error:
                    raise last_error

        if _MODEL is None:
            if last_error:
                raise last_error
            raise RuntimeError("Sentence transformer model could not be loaded.")

    return _MODEL


def _embedding_batch_size(default: int = 32) -> int:
    raw = os.getenv("EMBEDDING_BATCH_SIZE", str(default))
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def encode_with_fallback(
    texts: List[str],
    *,
    normalize_embeddings: bool = False,
    convert_to_numpy: bool = True,
    batch_size: Optional[int] = None,
):
    """Encode with recovery from meta-device and CUDA OOM failures."""
    global _MODEL, _MODEL_DEVICE
    effective_batch_size = batch_size or _embedding_batch_size()
    model = get_model()
    if model is None:
        raise RuntimeError("Sentence transformer model is unavailable after load attempts.")
    try:
        try:
            return model.encode(
                texts,
                batch_size=effective_batch_size,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=normalize_embeddings,
            )
        except TypeError as exc:
            if "batch_size" not in str(exc):
                raise
            return model.encode(
                texts,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=normalize_embeddings,
            )
    except Exception as exc:  # noqa: BLE001
        if _is_meta_tensor_error(exc) or (_is_cuda_oom(exc) and _MODEL_DEVICE != "cpu"):
            logging.error(
                "Embedding encode failed on device=%s (%s); retrying on cpu with smaller batch size",
                _MODEL_DEVICE,
                exc,
            )
            safe_batch_size = min(effective_batch_size, 16)
            try:
                model = get_model(reload=True, device="cpu")
                _MODEL_DEVICE = "cpu"
                return model.encode(
                    texts,
                    batch_size=safe_batch_size,
                    convert_to_numpy=convert_to_numpy,
                    normalize_embeddings=normalize_embeddings,
                )
            except Exception as retry_exc:  # noqa: BLE001
                if not _is_meta_tensor_error(retry_exc):
                    raise
                logging.error("CPU retry still failed with meta tensor error: %s", retry_exc)
                for fallback_name in _embedding_candidates()[1:]:
                    try:
                        logging.warning("Trying fallback embedding model on cpu: %s", fallback_name)
                        model = _load_sentence_transformer(fallback_name, "cpu")
                        _MODEL = model
                        _MODEL_DEVICE = "cpu"
                        return model.encode(
                            texts,
                            batch_size=safe_batch_size,
                            convert_to_numpy=convert_to_numpy,
                            normalize_embeddings=normalize_embeddings,
                        )
                    except Exception as fallback_exc:  # noqa: BLE001
                        logging.warning(
                            "Fallback embedding model %s failed: %s",
                            fallback_name,
                            fallback_exc,
                        )
                        continue
                raise retry_exc
        raise


def get_qdrant_client():
    """Lazy init for Qdrant client."""
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is None:
        _QDRANT_CLIENT = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=120)
    return _QDRANT_CLIENT


def get_vector_store() -> QdrantVectorStore:
    """Shared vector store wrapper to centralize collection handling."""
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        _VECTOR_STORE = QdrantVectorStore(client=get_qdrant_client())
    return _VECTOR_STORE


def get_hash_vectorizer():
    """Return a stable hashing vectorizer for sparse keyword vectors."""
    global _HASH_VECTORIZER
    if _HASH_VECTORIZER is None:
        _HASH_VECTORIZER = HashingVectorizer(
            n_features=4096,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            stop_words="english",
        )
    return _HASH_VECTORIZER


def decrypt_data(encrypted_value: str, encryption_key=Config.Encryption.ENCRYPTION_KEY) -> str:
    """Decrypts data using AES CBC mode."""
    try:
        key = hashlib.scrypt(encryption_key.encode(), salt=b'salt', n=16384, r=8, p=1, dklen=32)
        iv_hex, encrypted = encrypted_value.split(':')
        iv = bytes.fromhex(iv_hex)
        encrypted_bytes = bytes.fromhex(encrypted)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted_text = cipher.decrypt(encrypted_bytes).rstrip(b"\x00").decode('utf-8')
        logging.info("Decryption successful.")
        return decrypted_text
    except Exception as e:
        logging.error(f"Decryption failed: {e}")
        return ""


def fileProcessor(content, file):
    """Processes different types of documents and extracts text or dataframe."""
    extracted_data = {}
    try:
        file_name = file.split('/')[-1]
        extractor = get_doc_extractor()
        if content:
            logging.info(f"Extracting File {file_name}")
            if file_name.endswith(".csv"):
                df = pd.read_csv(BytesIO(content))
                extracted_data[file_name] = extractor.extract_dataframe(df, sheet_name=file_name)
            elif file_name.endswith((".xlsx", ".xls")):
                sheets = pd.read_excel(BytesIO(content), sheet_name=None)
                for sheet_name, df in sheets.items():
                    key = f"{file_name}#{sheet_name}"
                    extracted_data[key] = extractor.extract_dataframe(df, sheet_name=sheet_name)
            elif file_name.endswith(".json"):
                extracted_data[file_name] = json.loads(content.decode("utf-8"))
            elif file_name.endswith(".pdf"):
                extracted_data[file_name] = extractor.extract_text_from_pdf(content, filename=file_name)
            elif file_name.endswith(".docx"):
                extracted_data[file_name] = extractor.extract_text_from_docx(content, filename=file_name)
            elif file_name.endswith((".pptx", ".ppt")):
                extracted_data[file_name] = extractor.extract_text_from_pptx(content, filename=file_name)
            elif file_name.endswith(".txt"):
                extracted_data[file_name] = extractor.extract_text_from_txt(content, filename=file_name)
            else:
                extracted_data[file_name] = extractor.extract_text_from_txt(content, filename=file_name)
        return extracted_data
    except Exception as e:
        logging.error(f"Error processing file {file}: {e}")
        return {}


def read_s3_file(s3, bucket, file_key):
    """Reads a file from S3."""
    try:
        logging.info(f"Reading S3 file: {file_key} from bucket: {bucket}")
        obj = s3.get_object(Bucket=bucket, Key=file_key)
        content = obj["Body"].read()
        logging.info("S3 file read successfully.")
        return content
    except Exception as e:
        logging.error(f"Error reading S3 file {file_key}: {e}")
        return None


def get_s3_client(AWS_ACCESS_KEY, AWS_SECRET_KEY, Region):
    """Returns an S3 client."""
    try:
        awsProfile = Config.AWS.PROFILE
        subprocess.run(["aws", "configure", "set", "aws_access_key_id", AWS_ACCESS_KEY, "--profile", awsProfile])
        subprocess.run(["aws", "configure", "set", "aws_secret_access_key", AWS_SECRET_KEY, "--profile", awsProfile])
        subprocess.run(["aws", "configure", "set", "region", Region, "--profile", awsProfile])

        return b3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=Region,
        )
    except Exception as e:
        logging.error(f"Error creating S3 client: {e}")
        return None


def get_s3_document_info(s3_uri):
    """Reads document content from S3 using a URI."""
    parsed_uri = urlparse(s3_uri)
    bucket_name = parsed_uri.netloc
    object_key = parsed_uri.path.lstrip('/')
    s3 = b3.client(
        "s3",
        aws_access_key_id=Config.AWS.ACCESS_KEY,
        aws_secret_access_key=Config.AWS.SECRET_KEY,
        region_name=Config.AWS.REGION,
    )
    try:
        content = read_s3_file(s3, bucket_name, object_key)
        return content
    except Exception as e:
        return {"Error": str(e)}


def update_training_status(document_id, status, error_msg=None):
    """Updates training status in MongoDB for a specific document."""
    try:
        filter_criteria = {"_id": ObjectId(document_id)}
        update_data = {"status": status}

        if error_msg:
            update_data["training_error"] = error_msg
            update_data["training_failed_at"] = time.time()
        else:
            update_data["trained_at"] = time.time()
            # Clear any previous error messages
            update_data["training_error"] = None

        update_operation = {"$set": update_data}
        collection = db[Config.MongoDB.DOCUMENTS]
        result = collection.update_one(filter_criteria, update_operation)

        if result.matched_count > 0:
            logging.info(f"Training status for document {document_id} updated to {status}")
            return {"status": "success"}
        else:
            logging.warning(f"No document found with ID {document_id}")
            return {"status": "not_found"}
    except Exception as e:
        logging.error(f"Error updating training status for {document_id}: {e}")
        return {"status": "error", "message": str(e)}


def update_extraction_metadata(
    document_id: str,
    subscription_id: Optional[str],
    pickle_path: Optional[str],
    extracted_hash: Optional[str],
) -> None:
    """Persist extraction metadata without storing full text."""
    try:
        if ObjectId.is_valid(str(document_id)):
            filter_criteria = {"_id": ObjectId(str(document_id))}
        else:
            filter_criteria = {"_id": str(document_id)}

        update_data = {
            "document_id": str(document_id),
            "subscription_id": str(subscription_id) if subscription_id else None,
            "extracted_pickle_path": pickle_path,
            "extraction_status": "completed",
            "extracted_hash": extracted_hash,
            "extracted_at": time.time(),
            "updated_at": time.time(),
        }
        if pickle_path:
            from src.storage.azure_blob_client import has_blob_credentials

            container_name = (
                getattr(Config.AzureBlob, "DOCUMENT_CONTAINER_NAME", "document-content")
                if has_blob_credentials()
                else os.getenv("DOCUMENT_CONTENT_DIR", "document-content")
            )
            update_data["blob_reference"] = {
                "container": container_name,
                "blob_name": Path(pickle_path).name,
            }
        collection = db[Config.MongoDB.DOCUMENTS]
        collection.update_one(filter_criteria, {"$set": update_data})
    except Exception as exc:  # noqa: BLE001
        logging.error(f"Error updating extraction metadata for {document_id}: {exc}")


def update_security_screening(document_id: str, report: Dict[str, Any], status: str) -> None:
    """Persist security screening results for audit/debugging."""
    try:
        if ObjectId.is_valid(str(document_id)):
            filter_criteria = {"_id": ObjectId(str(document_id))}
        else:
            filter_criteria = {"_id": str(document_id)}

        update_data = {
            "security_screening": report,
            "security_screening_status": status,
            "security_screened_at": time.time(),
            "updated_at": time.time(),
        }
        collection = db[Config.MongoDB.DOCUMENTS]
        collection.update_one(filter_criteria, {"$set": update_data})
    except Exception as exc:  # noqa: BLE001
        logging.error(f"Error updating security screening for {document_id}: {exc}")


def run_security_screening(document_id: str, extracted_payload: Optional[Any] = None) -> Dict[str, Any]:
    """Run mandatory security screening using extracted payload when provided."""
    from src.screening.security_service import SecurityScreeningService

    service = SecurityScreeningService()
    return service.screen_document(
        document_id,
        extracted_payload=extracted_payload,
        include_overall_score=True,
    )


def resolve_subscription_id(document_id: str, provided: Optional[str] = None) -> str:
    if provided and str(provided).strip().lower() != "default":
        return str(provided).strip()
    try:
        from src.screening import storage_adapter
        resolved = storage_adapter.get_document_subscription_id(document_id)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"subscription_id lookup failed for document_id={document_id}: {exc}") from exc
    if not resolved or str(resolved).strip().lower() == "default":
        raise ValueError(f"subscription_id missing for document_id={document_id}")
    return str(resolved).strip()


def resolve_profile_id(document_id: str, provided: Optional[str] = None) -> str:
    if provided and str(provided).strip():
        return str(provided).strip()
    try:
        collection = db[Config.MongoDB.DOCUMENTS]
        record = None
        if ObjectId.is_valid(str(document_id)):
            record = collection.find_one({"_id": ObjectId(str(document_id))})
        if not record:
            record = collection.find_one({"_id": str(document_id)})
        if record:
            value = record.get("profileId") or record.get("profile_id") or record.get("profile")
            if value:
                return str(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"profile_id lookup failed for document_id={document_id}: {exc}") from exc
    raise ValueError(f"profile_id missing for document_id={document_id}")


def update_pii_stats(document_id, masked_count, high_confidential, pii_items=None):
    """Persist PII masking stats for a document."""
    try:
        filter_criteria = {}
        if ObjectId.is_valid(str(document_id)):
            filter_criteria["_id"] = ObjectId(str(document_id))
        else:
            filter_criteria["_id"] = str(document_id)

        update_data = {
            "pii_masked_count": int(masked_count),
            "pii_high_confidential": bool(high_confidential),
            "pii_has_pii": bool(masked_count or (pii_items or [])),
            "pii_items": pii_items or [],
            "pii_last_updated": time.time()
        }
        collection = db[Config.MongoDB.DOCUMENTS]
        collection.update_one(filter_criteria, {"$set": update_data})
    except Exception as e:
        logging.error(f"Error updating PII stats for {document_id}: {e}")


def clear_legacy_vetting_metadata() -> None:
    """Remove deprecated vetting metadata from all documents."""
    try:
        collection = db[Config.MongoDB.DOCUMENTS]
        result = collection.update_many({"vettingPoints": {"$exists": True}}, {"$unset": {"vettingPoints": ""}})
        if result.modified_count:
            logging.info("Cleared legacy vetting metadata from %s documents.", result.modified_count)
        else:
            logging.info("Legacy vetting metadata not present; no cleanup needed.")
    except Exception as exc:  # noqa: BLE001
        logging.warning("Legacy vetting metadata cleanup skipped: %s", exc)


def get_pii_stats(document_id):
    """Retrieve PII masking stats for a document."""
    try:
        # Try by ObjectId, then string, then fallback field names
        candidates = []
        if ObjectId.is_valid(str(document_id)):
            candidates.append({"_id": ObjectId(str(document_id))})
        candidates.append({"_id": str(document_id)})
        candidates.append({"document_id": str(document_id)})

        doc = None
        for crit in candidates:
            doc = db[Config.MongoDB.DOCUMENTS].find_one(crit, {
                "pii_masked_count": 1,
                "pii_high_confidential": 1,
                "pii_has_pii": 1,
                "pii_items": 1,
                "status": 1,
                "name": 1,
                "subscriptionId": 1,
                "profile": 1,
            })
            if doc:
                break

        if not doc:
            return None

        return {
            "document_id": str(document_id),
            "name": doc.get("name"),
            "status": doc.get("status"),
            "subscription_id": doc.get("subscriptionId"),
            "profile_id": str(doc.get("profile")) if doc.get("profile") else None,
            "pii_masked_count": doc.get("pii_masked_count", 0),
            "pii_high_confidential": doc.get("pii_high_confidential", False),
            "pii_has_pii": doc.get("pii_has_pii", False),
            "pii_items": doc.get("pii_items", [])
        }
    except Exception as e:
        logging.error(f"Error fetching PII stats for {document_id}: {e}")
        return None


def get_azure_docs(files, *, document_id: Optional[str] = None):
    """Fetch documents from Azure Blob Storage."""
    azure_blob = get_azure_blob()
    container_name = getattr(Config.AzureBlob, "DOCUMENT_CONTAINER_NAME", "document-content")
    blob_name = normalize_blob_name(files, container_name=container_name)
    container_candidates = []

    if isinstance(files, str) and files.startswith("az://"):
        parsed = urlparse(files)
        raw_path = parsed.path.lstrip("/")
        path_parts = raw_path.split("/", 1) if raw_path else []
        if parsed.netloc and parsed.netloc != container_name:
            container_candidates.append(parsed.netloc)
        elif path_parts and path_parts[0] and path_parts[0] != container_name:
            container_candidates.append(path_parts[0])

    teams_container = getattr(Config.Teams, "BLOB_CONTAINER", "")
    local_prefixed_blob = None
    if teams_container and teams_container != container_name:
        if blob_name.startswith("local/"):
            container_candidates.append(teams_container)
        else:
            local_prefixed_blob = f"local/{blob_name}"
            container_candidates.append(teams_container)

    container_candidates.append(container_name)
    seen_containers = set()
    try:
        last_exc = None
        for candidate_container in container_candidates:
            if candidate_container in seen_containers:
                continue
            seen_containers.add(candidate_container)
            container_client = azure_blob.get_container_client(candidate_container)
            name_variants = [blob_name]
            if candidate_container == teams_container and local_prefixed_blob:
                name_variants.insert(0, local_prefixed_blob)
            for name_variant in name_variants:
                for candidate in iter_blob_name_candidates(name_variant):
                    blob_client = container_client.get_blob_client(blob=candidate)
                    blob_url = sanitize_blob_url(getattr(blob_client, "url", ""))
                    if blob_url:
                        logging.info("Downloading blob for document_id=%s url=%s", document_id, blob_url)
                    else:
                        logging.info(
                            "Downloading blob for document_id=%s container=%s blob=%s",
                            document_id,
                            candidate_container,
                            candidate,
                        )
                    try:
                        return blob_client.download_blob().readall()
                    except ResourceNotFoundError as exc:
                        last_exc = exc
                        continue
        if last_exc:
            raise last_exc
        raise ResourceNotFoundError(message="Blob name candidates empty", response=None)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        error = classify_blob_error(exc, document_id=document_id, blob_name=blob_name)
        error_code, request_id = extract_azure_error_details(exc)
        logging.error(
            "Blob download failed document_id=%s blob=%s error_type=%s error_code=%s request_id=%s",
            document_id,
            blob_name,
            error.__class__.__name__,
            error_code,
            request_id,
        )
        raise error from exc


'-------------------------------modified by maha/maria-----------------------'
'---------------Added a PII control per subscription in connectData function--------------------'


def connectData(documentConnection):
    """
    Critical fix: Uses EXACT filename matching to prevent document_id collision
    """
    dataDict = {}

    for k, v in documentConnection.items():
        docData = v['dataDict']
        connData = v['connDict']
        profileId = str(docData['profile'])
        docId = str(docData['_id'])
        telemetry = telemetry_store() if METRICS_V2_ENABLED else None

        subscription_candidate = (
            docData.get('subscriptionId')
            or docData.get('subscription_id')
            or docData.get('subscription')
            or (connData.get('subscriptionId') if isinstance(connData, dict) else None)
            or (connData.get('subscription') if isinstance(connData, dict) else None)
        )
        try:
            subscriptionId = resolve_subscription_id(docId, subscription_candidate)
        except Exception as exc:
            logging.error(f"Subscription resolution failed for document {docId}: {exc}")
            update_training_status(docId, 'TRAINING_FAILED', 'subscription_id missing')
            continue

        # Check PII setting for this subscription
        pii_masking_enabled = get_subscription_pii_setting(subscriptionId)
        logging.info(f"Document {docId} (Subscription {subscriptionId}): PII masking = {pii_masking_enabled}")

        allowed_statuses = {'UNDER_REVIEW', 'TRAINING_FAILED'}

        if docData.get('status') in allowed_statuses:
            try:
                logging.info(f"=" * 80)
                logging.info(f"Processing document {docId}: {docData.get('name', 'Unknown')}")
                logging.info(f"=" * 80)
                if telemetry:
                    try:
                        telemetry.record_metadata_quality(docId, docData, expected_fields=list(docData.keys()))
                    except Exception:
                        telemetry.increment("metadata_parse_failures_count")
                all_extracted_docs = {}

                if docData['type'] == 'S3':
                    # S3 processing (unchanged)
                    bkName = connData['s3_details']['bucketName']
                    region = connData['s3_details']['region']
                    ak = decrypt_data(connData['s3_details']['accessKey']).split('\x0c')[0].strip()
                    sk = decrypt_data(connData['s3_details']['secretKey']).split('\x08')[0].strip()
                    s3 = get_s3_client(ak, sk, region)

                    if not s3:
                        logging.error(f"Failed to create S3 client for document {docId}")
                        update_training_status(docId, 'TRAINING_FAILED', 'Failed to create S3 client')
                        continue

                    objs = s3.list_objects_v2(Bucket=bkName)
                    file = [obj['Key'] for obj in objs.get("Contents", []) if obj['Key'] == docData['name']]

                    if not file:
                        logging.error(f"File {docData['name']} not found in S3 bucket {bkName}")
                        update_training_status(docId, 'TRAINING_FAILED', 'File not found in S3')
                        continue

                    docContent = read_s3_file(s3, bkName, file[0])
                    if docContent is None:
                        logging.error(f"Failed to read S3 file for document {docId}")
                        update_training_status(docId, 'TRAINING_FAILED', 'Failed to read S3 file')
                        continue

                    extractedDoc = fileProcessor(docContent, file[0])
                    if not extractedDoc:
                        logging.error(f"Failed to extract content from document {docId}")
                        update_training_status(docId, 'TRAINING_FAILED', 'Content extraction failed')
                        continue

                    all_extracted_docs.update(extractedDoc)

                elif docData['type'] == 'LOCAL':
                    # ============================================
                    #  CRITICAL FIX: EXACT filename matching
                    # ============================================

                    doc_name = docData.get('name', '')

                    if not doc_name:
                        logging.error(f"No filename found for document {docId}")
                        update_training_status(docId, 'TRAINING_FAILED', 'No filename specified')
                        continue

                    # Get all files from connector
                    all_connector_files = connData.get('locations', [])

                    logging.info(f"Looking for EXACT match: '{doc_name}'")
                    logging.info(f"Connector has {len(all_connector_files)} files")

                    #  FIX: Use EXACT filename matching
                    matching_files = []

                    for file_path in all_connector_files:
                        # Extract just the filename from the full path
                        # Handle both 'az://container/local/filename' and 'local/filename'
                        if '/' in file_path:
                            file_name_only = file_path.split('/')[-1]
                        else:
                            file_name_only = file_path

                        #  EXACT match (case-sensitive)
                        if file_name_only == doc_name:
                            matching_files.append(file_path)
                            logging.info(f" EXACT MATCH: {file_path} matches {doc_name}")
                            break  # Stop after first exact match

                    # L If no exact match, DO NOT fallback to partial matching
                    if not matching_files:
                        logging.error(f"L NO EXACT MATCH for document {docId} (name: '{doc_name}')")
                        logging.error(f"Available files in connector:")
                        for f in all_connector_files:
                            file_only = f.split('/')[-1] if '/' in f else f
                            logging.error(f"  - {file_only} (full path: {f})")
                        update_training_status(docId, 'TRAINING_FAILED', f'Exact file match not found: {doc_name}')
                        continue

                    #  Should only have ONE match due to break statement
                    if len(matching_files) > 1:
                        logging.error(
                            f"L CRITICAL: Multiple exact matches for {doc_name}: {matching_files}"
                        )
                        logging.error("This should not happen with exact matching!")
                        update_training_status(docId, 'TRAINING_FAILED', 'Multiple file matches')
                        continue

                    file_path = matching_files[0]
                    logging.info(f" Document {docId} will process ONLY: {file_path}")

                    # Process ONLY this document's specific file
                    try:
                        # Extract the blob name after 'az://' prefix while preserving nested paths
                        file_key = normalize_blob_name(
                            file_path, container_name=Config.AzureBlob.DOCUMENT_CONTAINER_NAME
                        )
                        logging.info(f"Reading file: {file_key} for document {docId}")

                        docContent = get_azure_docs(file_key, document_id=docId)
                        if docContent is None:
                            logging.error(f"Failed to read Azure file {file_key} for document {docId}")
                            update_training_status(docId, 'TRAINING_FAILED', f'Failed to read file {file_key}')
                            continue

                        extractedDoc = fileProcessor(docContent, file_path)
                        if not extractedDoc:
                            logging.error(f"Failed to extract content from file {file_key}")
                            update_training_status(docId, 'TRAINING_FAILED', 'Content extraction failed')
                            continue

                        all_extracted_docs.update(extractedDoc)

                        #  VERIFICATION: Should only have ONE file extracted
                        if len(all_extracted_docs) != 1:
                            logging.warning(
                                f"� Expected 1 extracted file for {docId}, got {len(all_extracted_docs)}: "
                                f"{list(all_extracted_docs.keys())}"
                            )
                        else:
                            logging.info(f" Successfully extracted 1 file from {file_key}")

                    except Exception as file_error:
                        logging.error(f"Error processing file {file_path}: {file_error}")
                        update_training_status(docId, 'TRAINING_FAILED', str(file_error))
                        continue

                # Apply PII masking
                if all_extracted_docs:
                    try:
                        save_info = save_extracted_pickle(docId, all_extracted_docs)
                        update_extraction_metadata(
                            docId,
                            subscriptionId,
                            save_info.get("path"),
                            save_info.get("sha256"),
                        )
                    except Exception as exc:
                        logging.error(f"Failed to persist extracted pickle for {docId}: {exc}")
                        update_training_status(docId, 'TRAINING_FAILED', 'Failed to persist extracted content')
                        continue

                    try:
                        security_report = run_security_screening(docId)
                        security_status = "passed"
                        risk_level = str(
                            security_report.get("overall_risk_level") or security_report.get("risk_level") or ""
                        ).upper()
                        if risk_level in {"HIGH", "CRITICAL"}:
                            security_status = "failed"
                        update_security_screening(docId, security_report, security_status)
                        if security_status != "passed":
                            logging.error(f"Security screening failed for document {docId}; blocking training")
                            update_training_status(docId, 'TRAINING_BLOCKED_SECURITY', 'Security screening failed')
                            continue
                    except Exception as exc:
                        logging.error(f"Security screening failed for {docId}: {exc}")
                        update_training_status(docId, 'TRAINING_FAILED', 'Security screening failed')
                        continue

                    if pii_masking_enabled:
                        masked_docs, pii_count, _high_conf, pii_items = mask_document_content(all_extracted_docs)
                        update_pii_stats(docId, pii_count, False, pii_items)

                        # Recompute embeddings for structured data after masking
                        for fname, content in masked_docs.items():
                            if isinstance(content, dict) and "texts" in content:
                                texts = content.get("texts") or []
                                if texts:
                                    content["embeddings"] = encode_with_fallback(
                                        texts,
                                        convert_to_numpy=True,
                                        normalize_embeddings=False,
                                    )
                                masked_docs[fname] = content
                    else:
                        logging.info(f"PII masking disabled for subscription {subscriptionId}")
                        masked_docs = all_extracted_docs
                        pii_count = 0
                        pii_items = []
                        update_pii_stats(docId, 0, False, [])

                    #  Store with explicit documentId
                    dataDict[docId] = {
                        'subscriptionId': subscriptionId,
                        'profileId': profileId,
                        'documentId': docId,  #  Explicit document_id
                        'extractedDoc': masked_docs,
                        'docName': docData.get('name', 'Unknown'),
                        'security': security_report,
                        'security_status': security_status,
                    }

                    logging.info(
                        f" Stored document {docId} with {len(masked_docs)} file(s) "
                        f"(PII masked: {pii_count})"
                    )
                else:
                    logging.error(f"No documents extracted for {docId}")
                    update_training_status(docId, 'TRAINING_FAILED', 'No content extracted')

            except Exception as e:
                logging.error(f"Error processing document {docId} ({docData.get('name', 'Unknown')}): {e}")
                update_training_status(docId, 'TRAINING_FAILED', str(e))

        elif docData['status'] == 'DELETED':
            profileData = str(docData['profile'])
            delete_embeddings(subscriptionId, profileData, docId)

    logging.info(f"=" * 80)
    logging.info(f" connectData completed: {len(dataDict)} documents processed")
    logging.info(f"=" * 80)

    return dataDict



def collectionConnect(name, mongo_db=None):
    """Fetches documents from a MongoDB collection."""
    logging.info(f"Fetching connection details for collection: {name}")
    try:
        collection = _resolve_mongo_db(mongo_db)[name]
        try:
            # Count documents to make emptiness explicit in the logs
            count = collection.count_documents({})
            logging.info(f"Collection '{name}' document count: {count}")
        except Exception as count_exc:
            logging.warning(f"Unable to count documents for collection '{name}': {count_exc}")
        return collection.find()
    except Exception as e:
        logging.error(f"Error connecting to collection {name}: {e}")
        raise


def extract_document_info(mongo_db=None):
    """Retrieves connector details from MongoDB."""
    try:
        db_handle = _resolve_mongo_db(mongo_db)
        logging.info(
            f"Extracting document info from DB: {Config.MongoDB.DB}, collections: {Config.MongoDB.DOCUMENTS}, {Config.MongoDB.CONNECTOR}")
        try:
            if not hasattr(db_handle, "list_collection_names") or not hasattr(db_handle, "__getitem__"):
                raise RuntimeError("DB not ready / invalid db handle")
            existing = db_handle.list_collection_names()
            logging.info(f"Existing collections in DB '{Config.MongoDB.DB}': {existing}")
        except Exception as lc_exc:
            logging.error("DB not ready / invalid db handle: %s", lc_exc)
            raise RuntimeError("DB not ready / invalid db handle") from lc_exc

        docs = collectionConnect(Config.MongoDB.DOCUMENTS, mongo_db=db_handle)
        Docs = {}
        # If docs is a cursor/iterable, iterate, otherwise it's probably an empty list
        for doc in docs:
            try:
                refConnector = doc.get('_id').__str__()
            except Exception:
                # fallback if doc['_id'] is not present
                refConnector = str(doc.get('_id', 'unknown'))
            Docs[refConnector] = doc
        logging.info(f"Found {len(Docs)} document definitions in collection '{Config.MongoDB.DOCUMENTS}'")

        connInfo = {}
        connectors = collectionConnect(Config.MongoDB.CONNECTOR, mongo_db=db_handle)
        for conn in connectors:
            try:
                connId = conn.get('_id').__str__()
            except Exception:
                connId = str(conn.get('_id', 'unknown'))
            connInfo[connId] = conn
        logging.info(f"Found {len(connInfo)} connector definitions in collection '{Config.MongoDB.CONNECTOR}'")

        connList = list(connInfo.keys())
        docInfo = {}
        for docId, docData in Docs.items():
            # docData expected to have a 'connector' field referencing a connector id
            try:
                connRef = docData.get('connector').__str__()
            except Exception:
                connRef = str(docData.get('connector', ''))
            if connRef in connList:
                docInfo[docId] = {'dataDict': docData, 'connDict': connInfo[connRef]}
            else:
                logging.info(f"Document {docId} references connector {connRef} which is not in connectors list")

        logging.info(f"Extracted {len(docInfo)} documents with valid connectors")
        return docInfo

    except Exception as e:
        logging.error(f"Error fetching connection details: {e}")
        raise


from qdrant_client.models import Filter, FieldCondition, MatchValue


def delete_embeddings(subscription_id: str, profile_id: str, document_id: str):
    """
    Delete all embeddings for a specific document from Qdrant.

    Args:
        subscription_id: Subscription ID (used to build collection name)
        profile_id: Profile ID for strict scoping
        document_id: Document ID to delete

    Returns:
        Dict with status and details
    """
    try:
        logging.info(
            f"[DELETE_EMBEDDINGS] Deleting embeddings for document_id={document_id}, "
            f"subscription_id={subscription_id}, profile_id={profile_id}"
        )

        if not profile_id:
            raise ValueError("profile_id is required for deleting embeddings")

        store = get_vector_store()
        result = store.delete_document(subscription_id, profile_id, document_id)
        return result

    except Exception as e:
        logging.error("[DELETE_EMBEDDINGS] Failed", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "document_id": document_id,
        }


def ensure_qdrant_collection(collection_name: str, vector_size: int) -> None:
    """Ensure Qdrant collection exists with multi-vector schema and payload indexes."""
    try:
        store = get_vector_store()
        store.ensure_collection(collection_name, vector_size)
    except Exception as e:
        logging.error(f"Error ensuring collection in Qdrant: {e}")
        raise


def save_embeddings_to_qdrant(
    embeddings: Dict,
    subscription_id: str,
    profile_id: str,
    doctag: str,
    source_filename: str,
    batch_size: int = 100,
):
    """Persist embeddings with deterministic chunk IDs and strict scoping."""
    try:
        if not subscription_id or str(subscription_id).strip().lower() == "default":
            raise ValueError("subscription_id is required and cannot be 'default' for embedding")
        if not profile_id:
            raise ValueError("profile_id is required for saving embeddings")

        if "embeddings" not in embeddings or embeddings["embeddings"] is None:
            logging.error(f"Error: 'embeddings' key missing or None for document {doctag}!")
            raise ValueError("Embeddings data is invalid")

        raw_vectors = embeddings["embeddings"]

        # Fallback: if embeddings are actually plain text strings, re-embed them
        if isinstance(raw_vectors, (list, tuple)) and raw_vectors and all(isinstance(v, str) for v in raw_vectors):
            logging.warning(
                f"Embeddings payload appears to be text; re-encoding {len(raw_vectors)} chunks for document {doctag}"
            )
            raw_vectors = encode_with_fallback(
                list(raw_vectors),
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

        normalized_vectors, vector_size = normalize_embedding_matrix(raw_vectors)
        if vector_size == 0:
            raise ValueError("Empty embeddings array")
        expected_dim = getattr(Config.Model, "EMBEDDING_DIM", None)
        if expected_dim and vector_size != expected_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {expected_dim}, got {vector_size}")

        texts = embeddings.get("texts") or []
        chunk_metadata = embeddings.get("chunk_metadata", []) or []
        pages = embeddings.get("pages") or []
        sections = embeddings.get("sections") or []
        summaries = embeddings.get("summaries") or []
        doc_type = embeddings.get("doc_type")
        ocr_confidence = embeddings.get("ocr_confidence")
        sparse_vectors = embeddings.get("sparse_vectors") or []
        doc_metadata = embeddings.get("doc_metadata") or {}

        filename = _safe_basename(doc_metadata.get("filename") or source_filename) or _safe_basename(source_filename)
        languages = _coerce_list(doc_metadata.get("languages"))
        products_name = doc_metadata.get("products_name")
        document_type = doc_metadata.get("document_type")
        profile_name = doc_metadata.get("profile_name")
        description = doc_metadata.get("description")
        doc_type = _pick_doc_type(doc_type, doc_metadata.get("doc_type"), doc_metadata.get("document_type"))
        document_type = _pick_doc_type(document_type, doc_metadata.get("document_type"), doc_metadata.get("doc_type"), doc_type)
        if doc_type and document_type:
            if doc_type.lower() != document_type.lower():
                logging.warning(
                    "Embedding metadata doc_type mismatch for %s: doc_type=%s document_type=%s; using document_type",
                    doctag,
                    doc_type,
                    document_type,
                )
                doc_type = document_type
            elif doc_type != document_type:
                doc_type = document_type

        if not texts:
            raise ValueError(f"No texts found for document {doctag}")

        max_len = min(len(texts), len(normalized_vectors))
        if max_len == 0:
            raise ValueError(f"No valid vectors for document {doctag}")

        if not sparse_vectors:
            sparse_vectors = build_sparse_vectors(texts)

        # Verify document_id in metadata
        if chunk_metadata:
            doc_ids_in_chunks = {meta.get("document_id") for meta in chunk_metadata if meta.get("document_id")}
            if len(doc_ids_in_chunks) != 1 or doctag not in doc_ids_in_chunks:
                raise ValueError(f"Chunk metadata must contain only document_id {doctag}")

        collection_name = build_collection_name(subscription_id)
        ensure_qdrant_collection(collection_name, vector_size)

        records: List[ChunkRecord] = []
        for idx in range(max_len):
            vector = normalized_vectors[idx]
            text = texts[idx]
            if not vector:
                continue

            chunk_meta = chunk_metadata[idx] if idx < len(chunk_metadata) else {}
            chunk_document_id = chunk_meta.get("document_id", str(doctag))
            if chunk_document_id != str(doctag):
                raise ValueError(f"Chunk {idx} document_id mismatch: {chunk_document_id} != {doctag}")

            chunk_id = chunk_meta.get(
                "chunk_id",
                compute_chunk_id(subscription_id, profile_id, doctag, source_filename, idx, text),
            )

            page_val = pages[idx] if idx < len(pages) else chunk_meta.get("page_number")
            section_val = chunk_meta.get("section_title", sections[idx] if idx < len(sections) else "")
            summary_val = summaries[idx] if idx < len(summaries) else chunk_meta.get("summary")
            section_path = chunk_meta.get("section_path") or section_val or "Untitled Section"
            page_start = chunk_meta.get("page_start", page_val)
            page_end = chunk_meta.get("page_end", page_val)
            chunk_char_len = len(text)
            chunk_hash = hashlib.sha256(str(text).encode("utf-8")).hexdigest()

            sparse_vector = None
            if idx < len(sparse_vectors):
                sv = sparse_vectors[idx]
                if sv.get("indices") and sv.get("values"):
                    sparse_vector = sv

            chunk_kind = chunk_meta.get("chunk_kind")
            if not chunk_kind:
                chunk_type = chunk_meta.get("chunk_type", "text")
                if chunk_type in {"table", "table_row", "table_header"}:
                    chunk_kind = "table_text"
                elif chunk_type == "image_caption":
                    chunk_kind = "image_caption"
                elif chunk_type == "summary":
                    chunk_kind = "section_summary"
                else:
                    chunk_kind = "section_text"

            evidence_pointer = None
            if section_val or page_start is not None or page_end is not None:
                if page_start is None and page_end is None:
                    page_range = "N/A"
                elif page_end is None or page_end == page_start:
                    page_range = str(page_start)
                else:
                    page_range = f"{page_start}-{page_end}"
                evidence_pointer = f"Section: {section_val or 'Section'}, Page: {page_range}"

            chunk_doc_type = _pick_doc_type(chunk_meta.get("doc_type"), doc_type, doc_metadata.get("doc_type"))
            chunk_document_type = _pick_doc_type(
                chunk_meta.get("document_type"),
                document_type,
                doc_metadata.get("document_type"),
                chunk_doc_type,
            )
            if chunk_doc_type and chunk_document_type:
                if chunk_doc_type.lower() != chunk_document_type.lower():
                    logging.warning(
                        "Chunk metadata doc_type mismatch for %s chunk %s: doc_type=%s document_type=%s; using document_type",
                        doctag,
                        idx,
                        chunk_doc_type,
                        chunk_document_type,
                    )
                    chunk_doc_type = chunk_document_type
                elif chunk_doc_type != chunk_document_type:
                    chunk_doc_type = chunk_document_type

            payload = {
                "subscription_id": str(subscription_id),
                "profile_id": str(profile_id),
                "profile_name": profile_name,
                "text": text,
                "document_id": str(doctag),
                "source_file": source_filename,
                "source_uri": chunk_meta.get("source_uri") or doc_metadata.get("source_uri") or source_filename,
                "filename": filename,
                "file_name": filename,
                "document_name": filename,
                "chunk_index": idx,
                "chunk_count": max_len,
                "page": page_val,
                "page_start": page_start,
                "page_end": page_end,
                "section_title": section_val or chunk_meta.get("section"),
                "section": section_val or chunk_meta.get("section"),
                "section_path": section_path,
                "summary": summary_val,
                "document_type": chunk_document_type or chunk_doc_type,
                "languages": chunk_meta.get("languages") or languages,
                "products_name": chunk_meta.get("products_name") or products_name,
                "description": chunk_meta.get("description") or description,
                "screening_status": chunk_meta.get("screening_status") or doc_metadata.get("screening_status"),
                "extraction_timestamp": (
                    chunk_meta.get("extraction_timestamp")
                    or doc_metadata.get("extraction_timestamp")
                    or doc_metadata.get("extracted_at")
                ),
                "ocr_confidence": chunk_meta.get("ocr_confidence") or ocr_confidence,
                "chunk_char_len": chunk_meta.get("chunk_char_len") or chunk_char_len,
                "chunk_hash": chunk_meta.get("chunk_hash") or chunk_hash,
                "text_hash": chunk_meta.get("text_hash") or chunk_hash,
                "chunk_sentence_complete": bool(chunk_meta.get("sentence_complete", False)),
                "chunk_id": chunk_id,
                "prev_chunk_id": chunk_meta.get("prev_chunk_id"),
                "next_chunk_id": chunk_meta.get("next_chunk_id"),
                "chunk_type": chunk_meta.get("chunk_type", "text"),
                "chunk_kind": chunk_kind,
                "section_id": chunk_meta.get("section_id"),
                "evidence_pointer": evidence_pointer,
            }

            try:
                payload = normalize_payload_metadata(payload, strict=True)
            except MetadataNormalizationError as exc:
                logging.error("Metadata normalization failed for chunk %s: %s", idx, exc)
                raise

            records.append(
                ChunkRecord(
                    chunk_id=str(chunk_id),
                    dense_vector=[float(x) for x in vector],
                    sparse_vector=sparse_vector,
                    payload=payload,
                )
            )

        if not records:
            raise ValueError(f"No valid embedding records generated for document {doctag}")

        saved = get_vector_store().upsert_records(collection_name, records, batch_size=batch_size)
        logging.info(
            f"Saved {saved} embeddings for document {doctag} in collection {collection_name} (profile={profile_id})"
        )
        return {"status": "success", "points_saved": saved}

    except Exception as e:
        logging.error(f"Error saving embeddings to Qdrant for document {doctag}, file {source_filename}: {e}")
        raise


# from enhanced_retrieval import chunk_text_for_embedding
# def train_on_document(text, subscription_id, profile_tag, doc_tag, doc_name):
#     """Trains and stores embeddings with enhanced chunking."""
#     try:
#         logging.info(f"Starting training for {doc_name}")
#
#         if isinstance(text, dict):
#             # Handle pre-embedded structured data (CSV/Excel)
#             result = save_embeddings_to_qdrant(
#                 text, subscription_id, profile_tag, doc_tag, doc_name
#             )
#             return f"Stored {result.get('points_saved', 0)} embeddings"
#
#         elif isinstance(text, str):
#             if not text.strip():
#                 raise ValueError(f"Empty content in {doc_name}")
#
#             # NEW: Use enhanced semantic chunking
#             chunks_with_meta = chunk_text_for_embedding(text, doc_name)
#
#             if not chunks_with_meta:
#                 raise ValueError(f"No valid chunks in {doc_name}")
#
#             # Extract chunks and metadata
#             chunks = [chunk_text for chunk_text, meta in chunks_with_meta]
#             chunk_metadata = [meta for chunk_text, meta in chunks_with_meta]
#
#             logging.info(f"Created {len(chunks)} enhanced chunks for {doc_name}")
#
#             # Generate embeddings
#             model = get_model()
#             embeddings_array = model.encode(
#                 chunks,
#                 convert_to_numpy=True,
#                 normalize_embeddings=True
#             )
#
#             # Build sparse vectors for keyword matching
#             from sklearn.feature_extraction.text import TfidfVectorizer
#             tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
#             tfidf_matrix = tfidf.fit_transform(chunks)
#
#             sparse_vectors = []
#             for row in tfidf_matrix:
#                 coo = row.tocoo()
#                 sparse_vectors.append({
#                     "indices": coo.col.tolist(),
#                     "values": coo.data.astype(np.float32).tolist()
#                 })
#
#             # Create summaries
#             summaries = [
#                 chunk[:200] + "..." if len(chunk) > 200 else chunk
#                 for chunk in chunks
#             ]
#
#             # Prepare embeddings dict with metadata
#             embeddings = {
#                 "embeddings": embeddings_array,
#                 "texts": chunks,
#                 "sparse_vectors": sparse_vectors,
#                 "summaries": summaries,
#                 "chunk_metadata": chunk_metadata  # NEW: Include metadata
#             }
#
#             # Save to Qdrant
#             result = save_embeddings_to_qdrant(
#                 embeddings,
#                 subscription_id,
#                 profile_tag,
#                 doc_tag,
#                 doc_name
#             )
#
#             return f"Stored {result.get('points_saved', 0)} embeddings"
#
#         else:
#             raise ValueError(f"Unsupported format: {type(text)}")
#
#     except Exception as e:
#         logging.error(f"Training error for {doc_name}: {e}")
#         raise


# CRITICAL FIX for train_on_document() in dataHandler.py

# Replace your existing train_on_document function with this:

from src.api.enhanced_retrieval import chunk_text_for_embedding, normalize_chunk_links


def _safe_basename(value: Optional[str]) -> str:
    if not value:
        return ""
    text = str(value)
    return text.split("/")[-1] if "/" in text else text


def _coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        parts = [part.strip() for part in re.split(r"[;,]", value) if part.strip()]
        return parts or [value.strip()]
    return [str(value)]


_SOURCE_TYPE_HINTS = {
    "LOCAL",
    "S3",
    "AZURE",
    "BLOB",
    "GCS",
    "GDRIVE",
    "GOOGLE_DRIVE",
    "ONEDRIVE",
    "DROPBOX",
    "BOX",
    "SHAREPOINT",
    "HTTP",
    "HTTPS",
    "URL",
    "FTP",
    "SFTP",
}


def _sanitize_doc_type(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.upper() in _SOURCE_TYPE_HINTS:
        return None
    return text


def _pick_doc_type(*candidates: Any) -> Optional[str]:
    for candidate in candidates:
        cleaned = _sanitize_doc_type(candidate)
        if cleaned:
            return cleaned
    return None


def _fetch_document_metadata(doc_tag: str, doc_name: str, doc_type_hint: Optional[str]) -> Dict[str, Any]:
    """Best-effort document metadata lookup for payload enrichment."""
    record: Dict[str, Any] = {}
    try:
        collection = db[Config.MongoDB.DOCUMENTS]
        if ObjectId.is_valid(str(doc_tag)):
            record = collection.find_one({"_id": ObjectId(str(doc_tag))}) or {}
        if not record:
            record = collection.find_one({"_id": str(doc_tag)}) or {}
        if not record:
            record = collection.find_one({"document_id": str(doc_tag)}) or {}
    except Exception as exc:  # noqa: BLE001
        logging.debug("Document metadata lookup failed for %s: %s", doc_tag, exc)
        record = {}

    metadata = record.get("metadata") or {}
    doc_type_hint = _sanitize_doc_type(doc_type_hint)
    doc_type = _pick_doc_type(
        record.get("doc_type"),
        record.get("document_type"),
        record.get("type"),
        metadata.get("doc_type"),
        metadata.get("document_type"),
        doc_type_hint,
    )
    languages = _coerce_list(record.get("languages") or record.get("language") or metadata.get("languages"))
    products_name = (
        record.get("products_name")
        or record.get("product_name")
        or record.get("product")
        or metadata.get("products_name")
        or metadata.get("product_name")
        or metadata.get("product")
    )
    document_type = _pick_doc_type(
        record.get("document_type"),
        record.get("doc_type"),
        metadata.get("document_type"),
        metadata.get("doc_type"),
        doc_type_hint,
        doc_type,
    )
    if doc_type and document_type:
        if doc_type.lower() != document_type.lower():
            logging.warning(
                "Document type conflict for %s: doc_type=%s document_type=%s; using document_type",
                doc_tag,
                doc_type,
                document_type,
            )
            doc_type = document_type
        elif doc_type != document_type:
            doc_type = document_type
    description = record.get("description") or record.get("summary") or metadata.get("description") or ""

    filename = _safe_basename(record.get("name") or metadata.get("name") or doc_name)
    profile_name = (
        record.get("profile_name")
        or record.get("profileName")
        or metadata.get("profile_name")
        or metadata.get("profileName")
    )
    if not profile_name:
        subscription_id = record.get("subscription_id") or record.get("subscriptionId") or metadata.get("subscription_id")
        profile_id = record.get("profile_id") or record.get("profileId") or metadata.get("profile_id")
        if subscription_id and profile_id:
            try:
                from src.profiles.profile_store import resolve_profile_name

                profile_name = resolve_profile_name(subscription_id=str(subscription_id), profile_id=str(profile_id))
            except Exception as exc:  # noqa: BLE001
                logging.debug("Profile name lookup failed for %s: %s", doc_tag, exc)

    return {
        "filename": filename or _safe_basename(doc_name),
        "doc_type": str(doc_type) if doc_type else None,
        "languages": languages,
        "products_name": str(products_name) if products_name else None,
        "document_type": str(document_type or doc_type) if (document_type or doc_type) else None,
        "description": str(description) if description else None,
        "profile_name": str(profile_name) if profile_name else None,
    }


def _maybe_update_document_fields(doc_tag: str, doc_metadata: Dict[str, Any]) -> None:
    update_fields: Dict[str, Any] = {}
    for key in ("profile_name", "products_name", "description", "document_type"):
        value = doc_metadata.get(key)
        if value is None or str(value).strip() == "":
            continue
        update_fields[key] = value
        update_fields[f"metadata.{key}"] = value
    if not update_fields:
        return
    try:
        from src.api.document_status import update_document_fields

        update_document_fields(doc_tag, update_fields)
    except Exception as exc:  # noqa: BLE001
        logging.debug("Document metadata update skipped for %s: %s", doc_tag, exc)


def _record_chunking_metrics(
    *,
    metrics_store: Any,
    doc_tag: str,
    chunk_lengths: List[int],
    sentence_complete_flags: List[bool],
) -> None:
    if not getattr(metrics_store, "available", False):
        return
    if not chunk_lengths:
        return
    avg_chars = float(sum(chunk_lengths) / max(1, len(chunk_lengths)))
    pct_complete = float(sum(1 for flag in sentence_complete_flags if flag) / max(1, len(sentence_complete_flags)))
    metrics_store.record(
        counters={"chunks_created": len(chunk_lengths)},
        values={
            "avg_chunk_chars": avg_chars,
            "pct_chunks_sentence_complete": pct_complete,
        },
        document_id=doc_tag,
        agent="embedding",
    )


def _score_chunk_quality(text: str) -> Tuple[float, Dict[str, float]]:
    cleaned = (text or "").strip()
    if not cleaned:
        return 0.0, {"length": 0.0, "alpha_ratio": 0.0, "symbol_ratio": 1.0, "repeat_ratio": 1.0}
    compact = re.sub(r"\s+", "", cleaned)
    length = len(cleaned)
    alpha_count = len(re.findall(r"[A-Za-z]", compact))
    symbol_count = len(re.findall(r"[^\w]", compact))
    line_tokens = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    repeat_ratio = 1.0
    if line_tokens:
        repeat_ratio = 1.0 - (len(set(line_tokens)) / max(1, len(line_tokens)))
    alpha_ratio = alpha_count / max(1, len(compact))
    symbol_ratio = symbol_count / max(1, len(compact))
    length_target = max(1, int(getattr(Config.Retrieval, "MIN_CHUNK_SIZE", 150)))
    length_score = min(1.0, length / max(1, length_target))
    quality = (0.4 * length_score) + (0.3 * alpha_ratio) + (0.2 * (1 - symbol_ratio)) + (0.1 * (1 - repeat_ratio))
    return round(max(0.0, min(1.0, quality)), 3), {
        "length": length_score,
        "alpha_ratio": alpha_ratio,
        "symbol_ratio": symbol_ratio,
        "repeat_ratio": repeat_ratio,
    }


def _apply_chunk_quality_filter(
    chunks: List[str],
    metadata: List[Dict[str, Any]],
) -> Tuple[List[str], List[Dict[str, Any]], int]:
    min_chars = int(getattr(Config.Retrieval, "MIN_CHUNK_CHARS", 40))
    min_quality = float(getattr(Config.Retrieval, "MIN_CHUNK_QUALITY", 0.2))
    max_symbol_ratio = float(getattr(Config.Retrieval, "MAX_SYMBOL_RATIO", 0.6))
    filtered_chunks: List[str] = []
    filtered_meta: List[Dict[str, Any]] = []
    dropped = 0
    for text, meta in zip(chunks, metadata):
        quality, breakdown = _score_chunk_quality(text)
        symbol_ratio = breakdown.get("symbol_ratio", 1.0)
        if len(text.strip()) < min_chars or quality < min_quality or symbol_ratio > max_symbol_ratio:
            dropped += 1
            continue
        meta["chunk_quality"] = quality
        filtered_chunks.append(text)
        filtered_meta.append(meta)
    for idx, meta in enumerate(filtered_meta):
        meta["chunk_index"] = idx
    return filtered_chunks, filtered_meta, dropped


def _extract_facts(text: str) -> List[str]:
    facts: List[str] = []
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if re.search(r"\d", candidate) or ":" in candidate:
            if len(candidate) <= 200:
                facts.append(candidate)
    return facts


def _table_rows_from_csv(table_csv: str, *, max_rows: int = 25) -> List[str]:
    rows: List[str] = []
    if not table_csv:
        return rows
    try:
        reader = csv.reader(StringIO(table_csv))
        headers = next(reader, [])
        for idx, row in enumerate(reader):
            if idx >= max_rows:
                break
            cells = [cell.strip() for cell in row]
            pairs = [f"{hdr}: {val}" for hdr, val in zip(headers, cells) if hdr.strip() and val.strip()]
            if pairs:
                rows.append("; ".join(pairs))
    except Exception:
        return rows
    return rows


def _build_chunk_metadata_from_section_chunks(
    section_chunks: List[Any],
    *,
    doc_tag: str,
    doc_type: Optional[str],
    doc_ocr_confidence: Optional[float],
) -> List[Dict[str, Any]]:
    metadata: List[Dict[str, Any]] = []
    for chunk in section_chunks:
        section_title = (getattr(chunk, "section_title", "") or "Untitled Section").strip() or "Untitled Section"
        section_path = (getattr(chunk, "section_path", "") or section_title).strip() or section_title
        page_start = getattr(chunk, "page_start", None)
        page_end = getattr(chunk, "page_end", None)
        chunk_index = int(getattr(chunk, "chunk_index", len(metadata)))
        sentence_complete = bool(getattr(chunk, "sentence_complete", False))
        metadata.append(
            {
                "document_id": doc_tag,
                "section_title": section_title,
                "section_path": section_path,
                "page_start": page_start,
                "page_end": page_end,
                "page_number": page_start,
                "chunk_index": chunk_index,
                "chunk_type": "text",
                "doc_type": doc_type,
                "ocr_confidence": doc_ocr_confidence,
                "sentence_complete": sentence_complete,
            }
        )
    return metadata


def _chunk_with_section_chunker(
    content: Any,
    *,
    doc_tag: str,
    doc_name: str,
    doc_type: Optional[str],
    doc_ocr_confidence: Optional[float],
) -> Tuple[List[str], List[Dict[str, Any]], Optional[float]]:
    chunker = SectionChunker()
    section_chunks = chunker.chunk_document(content, doc_internal_id=doc_tag, source_filename=doc_name)
    base_metadata = _build_chunk_metadata_from_section_chunks(
        section_chunks,
        doc_tag=doc_tag,
        doc_type=doc_type,
        doc_ocr_confidence=doc_ocr_confidence,
    )
    aligned_chunks: List[str] = []
    aligned_meta: List[Dict[str, Any]] = []
    for section_chunk, meta in zip(section_chunks, base_metadata):
        chunk_text = normalize_text(section_chunk.text)
        if not chunk_text:
            continue
        meta["chunk_index"] = len(aligned_chunks)
        aligned_chunks.append(chunk_text)
        aligned_meta.append(meta)

    aligned_chunks, aligned_meta, _dropped = _apply_chunk_quality_filter(aligned_chunks, aligned_meta)

    coverage_ratio = None
    if isinstance(content, ExtractedDocument) and content.full_text:
        full_text = normalize_text(content.full_text)
        if full_text:
            coverage_ratio = len("".join(aligned_chunks)) / max(1, len(full_text))
    return aligned_chunks, aligned_meta, coverage_ratio


def train_on_document(text, subscription_id, profile_id, doc_tag, doc_name):
    """
     COMPLETELY FIXED: Trains and stores embeddings with strict document_id verification

    Critical fixes:
    1. Always pass doc_tag as document_id parameter
    2. Verify chunks before generating embeddings
    3. Verify metadata before saving to Qdrant
    """
    try:
        if is_screening_stage():
            logging.warning("EMBED_BLOCKED_DURING_SCREENING: doc_id=%s", doc_tag)
            return {"status": "blocked", "chunks": 0, "points_saved": 0}
        telemetry = telemetry_store() if METRICS_V2_ENABLED else None
        metrics_store = get_metrics_store()
        logging.info(f"=" * 80)
        logging.info(f"Starting training for {doc_name}")
        logging.info(f"  Document ID: {doc_tag}")
        logging.info(f"  Subscription: {subscription_id}")
        logging.info(f"  Profile: {profile_id}")
        logging.info(f"=" * 80)

        if metrics_store.available:
            metrics_store.record(
                counters={"documents_processed": 1},
                document_id=doc_tag,
            )

        if not subscription_id or str(subscription_id).strip().lower() == "default":
            raise ValueError("subscription_id is required and cannot be 'default' for embedding")

        if not profile_id:
            raise ValueError("profile_id is required for training")

        if isinstance(text, dict):
            doc_type_hint = _pick_doc_type(text.get("doc_type"), text.get("document_type"), text.get("type"))
            doc_metadata = _fetch_document_metadata(doc_tag, doc_name, doc_type_hint)
            doc_type = doc_metadata.get("doc_type") or doc_type_hint
            _maybe_update_document_fields(doc_tag, doc_metadata)

            chunk_metadata = list(text.get("chunk_metadata") or [])
            texts = [str(t) for t in (text.get("texts") or []) if str(t).strip()]

            # If structured payload contains raw text but lacks chunking, chunk it safely.
            if not texts:
                raw_text = text.get("full_text") or text.get("text") or text.get("content")
                if isinstance(raw_text, str) and raw_text.strip():
                    texts, chunk_metadata, _ = _chunk_with_section_chunker(
                        raw_text,
                        doc_tag=doc_tag,
                        doc_name=doc_name,
                        doc_type=doc_type,
                        doc_ocr_confidence=None,
                    )

            if not texts:
                raise ValueError(f"No texts found for document {doc_tag}")

            # Ensure metadata exists and contains required section fields.
            if not chunk_metadata:
                chunk_metadata = [
                    {
                        "document_id": doc_tag,
                        "section_title": "Untitled Section",
                        "section_path": "Untitled Section",
                        "page_start": None,
                        "page_end": None,
                        "page_number": None,
                        "chunk_index": idx,
                        "chunk_type": "text",
                        "doc_type": doc_type,
                        "sentence_complete": str(texts[idx]).strip().endswith((".", "?", "!")),
                    }
                    for idx in range(len(texts))
                ]

            # Verify document_id consistency and backfill section metadata.
            doc_ids = {meta.get("document_id") for meta in chunk_metadata if meta.get("document_id")}
            doc_ids.discard(None)
            if len(doc_ids) > 1:
                raise ValueError(f"Multiple document_ids in structured data: {doc_ids}")
            if doc_ids and list(doc_ids)[0] != doc_tag:
                logging.warning(f"Fixing document_id in structured data: {doc_ids} -> {doc_tag}")
                for meta in chunk_metadata:
                    meta["document_id"] = doc_tag

            if len(chunk_metadata) < len(texts):
                for idx in range(len(chunk_metadata), len(texts)):
                    chunk_metadata.append(
                        {
                            "document_id": doc_tag,
                            "section_title": "Untitled Section",
                            "section_path": "Untitled Section",
                            "page_start": None,
                            "page_end": None,
                            "page_number": None,
                            "chunk_index": idx,
                            "chunk_type": "text",
                            "doc_type": doc_type,
                            "sentence_complete": str(texts[idx]).strip().endswith((".", "?", "!")),
                        }
                    )
            elif len(chunk_metadata) > len(texts):
                chunk_metadata = chunk_metadata[: len(texts)]

            for idx, meta in enumerate(chunk_metadata):
                section_title = (meta.get("section_title") or meta.get("section") or "Untitled Section").strip()
                if not section_title:
                    section_title = "Untitled Section"
                section_path = (meta.get("section_path") or section_title).strip() or section_title
                page_start = meta.get("page_start", meta.get("page_number"))
                page_end = meta.get("page_end", page_start)
                meta.update(
                    {
                        "document_id": doc_tag,
                        "section_title": section_title,
                        "section_path": section_path,
                        "page_start": page_start,
                        "page_end": page_end,
                        "page_number": page_start,
                        "chunk_index": idx,
                        "doc_type": meta.get("doc_type") or doc_type,
                        "chunk_kind": meta.get("chunk_kind", "section_text"),
                    }
                )
                if "sentence_complete" not in meta:
                    meta["sentence_complete"] = str(texts[idx]).strip().endswith((".", "?", "!"))

            texts, chunk_metadata, dropped_chunks = _apply_chunk_quality_filter(texts, chunk_metadata)
            if not texts:
                raise ValueError(f"Chunk quality filter removed all chunks for document {doc_tag}")

            chunk_lengths = [len(t) for t in texts]
            sentence_flags = [bool(meta.get("sentence_complete", False)) for meta in chunk_metadata]
            _record_chunking_metrics(
                metrics_store=metrics_store,
                doc_tag=doc_tag,
                chunk_lengths=chunk_lengths,
                sentence_complete_flags=sentence_flags,
            )

            text["texts"] = texts
            text["chunk_metadata"] = chunk_metadata
            text["doc_type"] = doc_type
            text["doc_metadata"] = doc_metadata

            expected_points = len(texts)
            result = save_embeddings_to_qdrant(text, subscription_id, profile_id, doc_tag, doc_name)
            saved = result.get("points_saved", 0)
            if expected_points and saved != expected_points:
                raise ValueError(
                    f"Embedding upsert mismatch for {doc_tag}: expected {expected_points}, saved {saved}"
                )
            logging.info(f" Stored {saved} structured embeddings")
            return {
                "status": "success",
                "points_saved": saved,
                "chunks": expected_points,
                "dropped_chunks": dropped_chunks,
                "coverage_ratio": None,
            }
        elif isinstance(text, ExtractedDocument):
            doc_type = text.doc_type
            ocr_confidences = (text.metrics or {}).get("ocr_confidences", []) if text.metrics else []
            doc_ocr_confidence = None
            if ocr_confidences:
                try:
                    doc_ocr_confidence = float(sum(ocr_confidences) / len(ocr_confidences))
                except Exception:
                    doc_ocr_confidence = None

            if metrics_store.available:
                candidates = text.chunk_candidates or []
                total_candidates = max(len(candidates), 1)
                short_chunks = sum(1 for cand in candidates if len((cand.text or "").strip()) < 20)
                missing_ratio = short_chunks / total_candidates
                text_accuracy = max(0.0, min(1.0, 1.0 - missing_ratio))
                structured_types = {"table", "table_row", "table_header", "ocr_text"}
                structured_chunks = sum(1 for cand in candidates if cand.chunk_type in structured_types)
                structure_ratio = structured_chunks / total_candidates

                error_pages = set()
                for err in text.errors or []:
                    match = re.search(r"page=(\d+)", str(err))
                    if match:
                        error_pages.add(int(match.group(1)))
                page_numbers = [cand.page for cand in candidates if cand.page is not None]
                total_pages = max(page_numbers, default=1)
                corrupted_ratio = len(error_pages) / max(total_pages, 1)

                metrics_store.record(
                    values={
                        "text_extraction_accuracy_pct": text_accuracy,
                        "structure_extraction_accuracy_pct": structure_ratio,
                        "missing_content_ratio": missing_ratio,
                        "corrupted_page_ratio": corrupted_ratio,
                    },
                    document_id=doc_tag,
                )

                for conf in ocr_confidences:
                    metrics_store.record(
                        values={"ocr_confidence": float(conf)},
                        minmax={"ocr_confidence": float(conf)},
                        document_id=doc_tag,
                    )

            doc_metadata = _fetch_document_metadata(doc_tag, doc_name, doc_type)
            doc_type = doc_metadata.get("doc_type") or doc_type
            _maybe_update_document_fields(doc_tag, doc_metadata)

            chunks, chunk_metadata, coverage_ratio = _chunk_with_section_chunker(
                text,
                doc_tag=doc_tag,
                doc_name=doc_name,
                doc_type=doc_type,
                doc_ocr_confidence=doc_ocr_confidence,
            )

            full_text = normalize_text(text.full_text or "")
            coverage_threshold = float(getattr(Config.Retrieval, "CHUNK_COVERAGE_THRESHOLD", 0.98))
            if full_text and coverage_ratio is not None and coverage_ratio < coverage_threshold:
                logging.warning(
                    "Section chunk coverage %.3f below threshold %.3f for %s; falling back to full text",
                    coverage_ratio,
                    coverage_threshold,
                    doc_name,
                )
                chunks, chunk_metadata, _ = _chunk_with_section_chunker(
                    full_text,
                    doc_tag=doc_tag,
                    doc_name=doc_name,
                    doc_type=doc_type,
                    doc_ocr_confidence=doc_ocr_confidence,
                )
                coverage_ratio = len("".join(chunks)) / max(1, len(full_text)) if full_text else coverage_ratio

            if not chunks:
                raise ValueError(f"No valid chunks extracted for {doc_name}")

            for meta in chunk_metadata:
                meta.setdefault("chunk_kind", "section_text")

            chunks, chunk_metadata, dropped_chunks = _apply_chunk_quality_filter(chunks, chunk_metadata)
            if not chunks:
                raise ValueError(f"Chunk quality filter removed all chunks for {doc_name}")

            chunk_lengths = [len(chunk) for chunk in chunks]
            sentence_flags = [bool(meta.get("sentence_complete", False)) for meta in chunk_metadata]
            _record_chunking_metrics(
                metrics_store=metrics_store,
                doc_tag=doc_tag,
                chunk_lengths=chunk_lengths,
                sentence_complete_flags=sentence_flags,
            )

            logging.info("Generated %s section-aware chunks for %s", len(chunks), doc_name)

            embed_start = time.time()
            embeddings_array = encode_with_fallback(
                chunks,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            embed_latency_ms = (time.time() - embed_start) * 1000
            if telemetry:
                telemetry.increment("embedding_requests_count")
                telemetry.increment("total_chunks_embedded", amount=len(chunks))
                telemetry.observe("embedding_latency_ms", embed_latency_ms)
                telemetry.record_doc_metric(doc_tag, "chunks_embedded", len(chunks))
                telemetry.set_gauge("last_embedding_time", time.time())
            if metrics_store.available and len(embeddings_array) > 0:
                if len(embeddings_array) > 1:
                    sims = np.sum(embeddings_array[:-1] * embeddings_array[1:], axis=1)
                    coherence = float(np.mean(sims))
                else:
                    coherence = 1.0
                drift = max(0.0, min(1.0, 1.0 - coherence))
                metrics_store.record(
                    values={
                        "embedding_coherence_score": coherence,
                        "chunk_semantic_drift_score": drift,
                    },
                    document_id=doc_tag,
                )
            sparse_vectors = build_sparse_vectors(chunks)
            summaries = compute_section_summaries(chunks, chunk_metadata, extracted=text)

            ctx = ContextUnderstanding()
            doc_summary_bundle = ctx.summarize_document(text)
            doc_summary_text = (doc_summary_bundle.get("abstract") or "").strip()
            section_summary_map = doc_summary_bundle.get("section_summaries") or {}
            section_title_to_pages = {
                (sec.title or "").strip(): (sec.start_page, sec.end_page) for sec in text.sections
            }

            extra_chunks: List[str] = []
            extra_meta: List[Dict[str, Any]] = []

            if doc_summary_text:
                extra_chunks.append(doc_summary_text)
                extra_meta.append(
                    {
                        "document_id": doc_tag,
                        "section_title": "Document Summary",
                        "section_path": "Document Summary",
                        "page_start": None,
                        "page_end": None,
                        "page_number": None,
                        "chunk_index": len(chunks) + len(extra_chunks) - 1,
                        "chunk_type": "summary",
                        "chunk_kind": "doc_summary",
                        "doc_type": doc_type,
                        "sentence_complete": True,
                    }
                )

            for title, summary in section_summary_map.items():
                summary_text = str(summary).strip()
                if not summary_text:
                    continue
                pages = section_title_to_pages.get((title or "").strip())
                page_start, page_end = (pages or (None, None))
                extra_chunks.append(summary_text)
                extra_meta.append(
                    {
                        "document_id": doc_tag,
                        "section_title": title or "Section",
                        "section_path": title or "Section",
                        "page_start": page_start,
                        "page_end": page_end,
                        "page_number": page_start,
                        "chunk_index": len(chunks) + len(extra_chunks) - 1,
                        "chunk_type": "summary",
                        "chunk_kind": "section_summary",
                        "doc_type": doc_type,
                        "sentence_complete": True,
                    }
                )

            for chunk_text, meta in zip(chunks, chunk_metadata):
                facts = _extract_facts(chunk_text)
                if len(facts) < 2:
                    continue
                fact_text = "\n".join(facts[:20])
                extra_chunks.append(fact_text)
                extra_meta.append(
                    {
                        "document_id": doc_tag,
                        "section_title": meta.get("section_title") or "Facts",
                        "section_path": meta.get("section_path") or meta.get("section_title") or "Facts",
                        "page_start": meta.get("page_start"),
                        "page_end": meta.get("page_end"),
                        "page_number": meta.get("page_start"),
                        "chunk_index": len(chunks) + len(extra_chunks) - 1,
                        "chunk_type": "fact",
                        "chunk_kind": "structured_field",
                        "doc_type": doc_type,
                        "sentence_complete": True,
                    }
                )

            for table in text.tables or []:
                table_text = normalize_text(table.csv or table.text or "")
                if table_text:
                    extra_chunks.append(table_text)
                    extra_meta.append(
                        {
                            "document_id": doc_tag,
                            "section_title": "Table",
                            "section_path": "Table",
                            "page_start": table.page,
                            "page_end": table.page,
                            "page_number": table.page,
                            "chunk_index": len(chunks) + len(extra_chunks) - 1,
                            "chunk_type": "table",
                            "chunk_kind": "table_text",
                            "doc_type": doc_type,
                            "sentence_complete": True,
                        }
                    )
                for row_text in _table_rows_from_csv(table.csv or ""):
                    extra_chunks.append(row_text)
                    extra_meta.append(
                        {
                            "document_id": doc_tag,
                            "section_title": "Table Row",
                            "section_path": "Table Row",
                            "page_start": table.page,
                            "page_end": table.page,
                            "page_number": table.page,
                            "chunk_index": len(chunks) + len(extra_chunks) - 1,
                            "chunk_type": "table_row",
                            "chunk_kind": "structured_field",
                            "doc_type": doc_type,
                            "sentence_complete": True,
                        }
                    )

            chunk_metadata = normalize_chunk_links(
                chunk_metadata,
                subscription_id=subscription_id,
                profile_id=profile_id,
                document_id=doc_tag,
                doc_name=doc_name,
                chunks=chunks,
            )
            for idx, meta in enumerate(chunk_metadata):
                section_path = meta.get("section_path") or meta.get("section_title") or "Untitled Section"
                meta["section_path"] = section_path
                meta["section_id"] = meta.get("section_id") or hashlib.sha1(
                    f"{doc_tag}|{section_path}".encode("utf-8")
                ).hexdigest()[:12]
                meta["chunk_type"] = meta.get("chunk_type", "text")
                meta["chunk_kind"] = meta.get("chunk_kind", "section_text")
                page_start = meta.get("page_start", meta.get("page_number"))
                page_end = meta.get("page_end", page_start)
                meta["page_start"] = page_start
                meta["page_end"] = page_end
                meta["page_number"] = page_start
                meta["chunk_char_len"] = meta.get("chunk_char_len") or len(chunks[idx])
                meta["chunk_hash"] = meta.get("chunk_hash") or hashlib.sha256(
                    chunks[idx].encode("utf-8")
                ).hexdigest()
                meta["sentence_complete"] = bool(meta.get("sentence_complete", chunks[idx].strip().endswith((".", "?", "!"))))

            if extra_chunks:
                extra_chunks, extra_meta, dropped_extra = _apply_chunk_quality_filter(extra_chunks, extra_meta)
                for idx, meta in enumerate(extra_meta):
                    meta["chunk_index"] = len(chunks) + idx
                dropped_chunks += dropped_extra
                chunks.extend(extra_chunks)
                chunk_metadata.extend(extra_meta)
                sparse_vectors.extend(build_sparse_vectors(extra_chunks))
                extra_embeddings = encode_with_fallback(
                    extra_chunks,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                embeddings_array = list(embeddings_array) + list(extra_embeddings)
                summaries.extend([None for _ in extra_chunks])

            embeddings_payload = {
                "embeddings": embeddings_array,
                "texts": chunks,
                "sparse_vectors": sparse_vectors,
                "summaries": summaries,
                "chunk_metadata": chunk_metadata,
                "pages": [m.get("page_number") for m in chunk_metadata],
                "sections": [m.get("section_title") for m in chunk_metadata],
                "doc_type": doc_type,
                "doc_metadata": doc_metadata,
                "ocr_confidence": doc_ocr_confidence,
            }

            result = save_embeddings_to_qdrant(embeddings_payload, subscription_id, profile_id, doc_tag, doc_name)
            saved = result.get("points_saved", 0)
            expected_points = len(chunks)
            if saved != expected_points:
                raise ValueError(
                    f"Embedding upsert mismatch for {doc_tag}: expected {expected_points}, saved {saved}"
                )
            logging.info("Stored %s section-aware embeddings for %s", saved, doc_name)
            return {
                "status": "success",
                "points_saved": saved,
                "chunks": expected_points,
                "dropped_chunks": dropped_chunks,
                "coverage_ratio": coverage_ratio,
            }

        elif isinstance(text, str):
            if not text.strip():
                raise ValueError(f"Empty content in {doc_name}")

            logging.info(f"Chunking document with document_id={doc_tag}")

            doc_metadata = _fetch_document_metadata(doc_tag, doc_name, None)
            doc_type = doc_metadata.get("doc_type")

            try:
                chunks, chunk_metadata, _ = _chunk_with_section_chunker(
                    text,
                    doc_tag=doc_tag,
                    doc_name=doc_name,
                    doc_type=doc_type,
                    doc_ocr_confidence=None,
                )
            except Exception as exc:  # noqa: BLE001
                # Fallback to the legacy chunker to avoid total failure on edge cases.
                logging.warning("Section chunking failed for %s: %s; falling back", doc_name, exc)
                chunks_with_meta = chunk_text_for_embedding(text, doc_name, document_id=doc_tag)
                chunks = [chunk_text for chunk_text, _meta in chunks_with_meta if (chunk_text or "").strip()]
                chunk_metadata = [meta for chunk_text, meta in chunks_with_meta if (chunk_text or "").strip()]

            if not chunks:
                raise ValueError(f"No valid chunks in {doc_name}")

            chunk_lengths = [len(chunk) for chunk in chunks]
            sentence_flags = [bool(meta.get("sentence_complete", False)) for meta in chunk_metadata]
            _record_chunking_metrics(
                metrics_store=metrics_store,
                doc_tag=doc_tag,
                chunk_lengths=chunk_lengths,
                sentence_complete_flags=sentence_flags,
            )

            chunks, chunk_metadata, dropped_chunks = _apply_chunk_quality_filter(chunks, chunk_metadata)
            if not chunks:
                raise ValueError(f"Chunk quality filter removed all chunks for {doc_name}")

            # Normalize chunk linkage and ids to avoid mismatched prev/next references
            chunk_metadata = normalize_chunk_links(
                chunk_metadata,
                subscription_id=subscription_id,
                profile_id=profile_id,
                document_id=doc_tag,
                doc_name=doc_name,
                chunks=chunks
            )

            fact_chunks: List[str] = []
            fact_meta: List[Dict[str, Any]] = []
            for chunk_text, meta in zip(chunks, chunk_metadata):
                facts = _extract_facts(chunk_text)
                if len(facts) < 2:
                    continue
                fact_text = "\n".join(facts[:20])
                fact_chunks.append(fact_text)
                fact_meta.append(
                    {
                        "document_id": doc_tag,
                        "section_title": meta.get("section_title") or "Facts",
                        "section_path": meta.get("section_path") or meta.get("section_title") or "Facts",
                        "page_start": meta.get("page_start"),
                        "page_end": meta.get("page_end"),
                        "page_number": meta.get("page_start"),
                        "chunk_index": len(chunks) + len(fact_chunks) - 1,
                        "chunk_type": "fact",
                        "chunk_kind": "structured_field",
                        "doc_type": doc_type,
                        "sentence_complete": True,
                    }
                )
            if fact_chunks:
                fact_chunks, fact_meta, dropped_facts = _apply_chunk_quality_filter(fact_chunks, fact_meta)
                for idx, meta in enumerate(fact_meta):
                    meta["chunk_index"] = len(chunks) + idx
                dropped_chunks += dropped_facts
                chunks.extend(fact_chunks)
                chunk_metadata.extend(fact_meta)

            # Generate embeddings
            logging.info(f"Generating embeddings for {len(chunks)} chunks")
            embed_start = time.time()
            embeddings_array = encode_with_fallback(
                chunks,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            embed_latency_ms = (time.time() - embed_start) * 1000
            if telemetry:
                telemetry.increment("embedding_requests_count")
                telemetry.increment("total_chunks_embedded", amount=len(chunks))
                telemetry.observe("embedding_latency_ms", embed_latency_ms)
                telemetry.record_doc_metric(doc_tag, "chunks_embedded", len(chunks))
                telemetry.set_gauge("last_embedding_time", time.time())

            sparse_vectors = build_sparse_vectors(chunks)

            # Create summaries
            summaries = compute_section_summaries(chunks, chunk_metadata)
            if not any(summaries):
                summaries = [
                    chunk[:200] + "..." if len(chunk) > 200 else chunk
                    for chunk in chunks
                ]

            #  VERIFICATION STEP 2: Final check before saving
            logging.info(f"Final verification before saving to Qdrant")
            for idx, meta in enumerate(chunk_metadata):
                if meta.get('document_id') != doc_tag:
                    raise ValueError(
                        f"L Chunk {idx} has wrong document_id: {meta.get('document_id')} "
                        f"(expected {doc_tag})"
                    )
                section_title = (meta.get("section_title") or "Untitled Section").strip() or "Untitled Section"
                section_path = (meta.get("section_path") or section_title).strip() or section_title
                meta["section_title"] = section_title
                meta["section_path"] = section_path
                meta["section_id"] = meta.get("section_id") or hashlib.sha1(
                    f"{doc_tag}|{section_path}".encode("utf-8")
                ).hexdigest()[:12]
                meta["chunk_type"] = meta.get("chunk_type", "text")
                meta["chunk_kind"] = meta.get("chunk_kind", "section_text")
                page_start = meta.get("page_start", meta.get("page_number"))
                page_end = meta.get("page_end", page_start)
                meta["page_start"] = page_start
                meta["page_end"] = page_end
                meta["page_number"] = page_start
                meta["chunk_char_len"] = meta.get("chunk_char_len") or len(chunks[idx])
                meta["chunk_hash"] = meta.get("chunk_hash") or hashlib.sha256(
                    chunks[idx].encode("utf-8")
                ).hexdigest()
                meta["sentence_complete"] = bool(meta.get("sentence_complete", chunks[idx].strip().endswith((".", "?", "!"))))
                # chunk_id/prev/next already normalized by normalize_chunk_links

            # Prepare embeddings dict with metadata
            embeddings = {
                "embeddings": embeddings_array,
                "texts": chunks,
                "sparse_vectors": sparse_vectors,
                "summaries": summaries,
                "chunk_metadata": chunk_metadata,
                "doc_metadata": doc_metadata,
                "doc_type": doc_type,
            }

            # Save to Qdrant (will perform additional verification)
            logging.info(f"Saving to Qdrant...")
            result = save_embeddings_to_qdrant(
                embeddings,
                subscription_id,
                profile_id,
                doc_tag,
                doc_name
            )

            logging.info(f"=" * 80)
            saved = result.get("points_saved", 0)
            expected_points = len(chunks)
            if saved != expected_points:
                raise ValueError(
                    f"Embedding upsert mismatch for {doc_tag}: expected {expected_points}, saved {saved}"
                )
            logging.info(f" SUCCESS: Stored {saved} embeddings")
            logging.info(f"  Document ID: {doc_tag}")
            logging.info(f"  File: {doc_name}")
            logging.info(f"=" * 80)

            return {
                "status": "success",
                "points_saved": saved,
                "chunks": expected_points,
                "dropped_chunks": dropped_chunks,
                "coverage_ratio": None,
            }

        else:
            raise ValueError(f"Unsupported format: {type(text)}")

    except Exception as e:
        logging.error(f"=" * 80)
        logging.error(f"L TRAINING FAILED for {doc_name}")
        logging.error(f"  Document ID: {doc_tag}")
        logging.error(f"  Error: {e}")
        logging.error(f"=" * 80)
        try:
            if telemetry:
                telemetry.increment("embedding_failures_count")
        except Exception:
            pass
        raise


def process_document_pipeline(
    document_id: str,
    file_bytes: bytes,
    filename: str,
    subscription_id: Optional[str] = None,
    profile_id: Optional[str] = None,
    embed_after: bool = False,
) -> Dict[str, Any]:
    """End-to-end ingestion pipeline for a single document."""
    errors: List[str] = []
    extraction_info: Dict[str, Any] = {"status": "failed"}
    security_info: Dict[str, Any] = {"status": "failed"}
    embedding_info: Dict[str, Any] = {"status": "skipped", "chunks": 0, "upserted": 0}
    cleanup_info: Dict[str, Any] = {"pickle_deleted": False, "cleanup_pending": True}
    save_info: Dict[str, Any] = {}

    try:
        extracted_doc = fileProcessor(file_bytes, filename)
        if not extracted_doc:
            raise ValueError("No content extracted from file")
        save_info = save_extracted_pickle(document_id, extracted_doc)
        update_extraction_metadata(
            document_id,
            subscription_id,
            save_info.get("path"),
            save_info.get("sha256"),
        )
        extraction_info = {"status": "ok", "blob": save_info.get("blob_name")}
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
        return {
            "document_id": document_id,
            "extraction": extraction_info,
            "security": security_info,
            "embedding": embedding_info,
            "cleanup": cleanup_info,
            "errors": errors,
        }

    try:
        subscription_id = resolve_subscription_id(document_id, subscription_id)
        profile_id = resolve_profile_id(document_id, profile_id)
        if save_info:
            update_extraction_metadata(
                document_id,
                subscription_id,
                save_info.get("path"),
                save_info.get("sha256"),
            )
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
        return {
            "document_id": document_id,
            "extraction": extraction_info,
            "security": security_info,
            "embedding": embedding_info,
            "cleanup": cleanup_info,
            "errors": errors,
        }

    try:
        security_report = run_security_screening(document_id)
        risk_level = str(security_report.get("overall_risk_level") or security_report.get("risk_level") or "").upper()
        security_status = "passed" if risk_level not in {"HIGH", "CRITICAL"} else "failed"
        security_info = {
            "status": security_status,
            "risk_level": security_report.get("risk_level"),
            "overall_risk_level": security_report.get("overall_risk_level"),
        }
        update_security_screening(document_id, security_report, security_status)
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
        security_info = {"status": "failed", "risk_level": "UNKNOWN"}
        return {
            "document_id": document_id,
            "extraction": extraction_info,
            "security": security_info,
            "embedding": embedding_info,
            "cleanup": cleanup_info,
            "errors": errors,
        }

    if security_info.get("status") != "passed":
        embedding_info = {"status": "skipped", "chunks": 0, "upserted": 0}
        return {
            "document_id": document_id,
            "extraction": extraction_info,
            "security": security_info,
            "embedding": embedding_info,
            "cleanup": cleanup_info,
            "errors": errors,
        }

    try:
        pii_masking_enabled = get_subscription_pii_setting(subscription_id)
        if pii_masking_enabled:
            masked_docs, pii_count, _high_conf, pii_items = mask_document_content(extracted_doc)
            update_pii_stats(document_id, pii_count, False, pii_items)
        else:
            masked_docs = extracted_doc
            update_pii_stats(document_id, 0, False, [])
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
        embedding_info = {"status": "failed", "chunks": 0, "upserted": 0}
        return {
            "document_id": document_id,
            "extraction": extraction_info,
            "security": security_info,
            "embedding": embedding_info,
            "cleanup": cleanup_info,
            "errors": errors,
        }

    if not embed_after:
        embedding_info = {"status": "pending", "chunks": 0, "upserted": 0}
        try:
            from src.api.document_status import update_document_fields
            from src.api.statuses import STATUS_SCREENING_COMPLETED

            update_document_fields(
                document_id,
                {"status": STATUS_SCREENING_COMPLETED, "updated_at": time.time()},
            )
        except Exception as exc:  # noqa: BLE001
            logging.debug("Failed to update screening status for %s: %s", document_id, exc)
        return {
            "document_id": document_id,
            "extraction": extraction_info,
            "security": security_info,
            "embedding": embedding_info,
            "cleanup": cleanup_info,
            "errors": errors,
        }

    try:
        total_chunks = 0
        total_upserted = 0
        for file_name, file_content in masked_docs.items():
            result = train_on_document(
                file_content,
                subscription_id,
                profile_id,
                document_id,
                file_name,
            )
            total_chunks += result.get("chunks", 0)
            total_upserted += result.get("points_saved", 0)
        if total_chunks != total_upserted:
            raise ValueError(f"Embedding upsert mismatch: expected {total_chunks}, saved {total_upserted}")
        embedding_info = {"status": "completed", "chunks": total_chunks, "upserted": total_upserted}
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
        embedding_info = {"status": "failed", "chunks": 0, "upserted": 0}
        return {
            "document_id": document_id,
            "extraction": extraction_info,
            "security": security_info,
            "embedding": embedding_info,
            "cleanup": cleanup_info,
            "errors": errors,
        }

    try:
        cleanup_info["pickle_deleted"] = delete_extracted_pickle(document_id)
        cleanup_info["cleanup_pending"] = not cleanup_info["pickle_deleted"]
    except Exception as exc:  # noqa: BLE001
        logging.warning(f"Cleanup failed for {document_id}: {exc}")
        cleanup_info["pickle_deleted"] = False
        cleanup_info["cleanup_pending"] = True

    return {
        "document_id": document_id,
        "extraction": extraction_info,
        "security": security_info,
        "embedding": embedding_info,
        "cleanup": cleanup_info,
        "errors": errors,
    }


def trainData(mongo_db=None):
    """Extraction-only pipeline for documents eligible for processing."""
    try:
        from src.api.extraction_service import extract_documents

        logging.info("=" * 80)
        logging.info("Starting extraction process")
        logging.info("=" * 80)
        return extract_documents(mongo_db=mongo_db)
    except MongoProviderError:
        raise
    except RuntimeError:
        raise
    except Exception as e:
        logging.error(f"Critical error in extraction data: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "results": None}


# New function: train_single_document
def train_single_document(doc_id: str, mongo_db=None):
    """Extract a single document identified by its string ID."""
    try:
        from src.api.extraction_service import extract_single_document

        logging.info(f"Starting single-document extraction for ID: {doc_id}")
        return extract_single_document(doc_id, mongo_db=mongo_db)
    except MongoProviderError:
        raise
    except RuntimeError:
        raise
    except Exception as e:
        logging.error(f"Critical error during single-document extraction for {doc_id}: {e}", exc_info=True)
        update_training_status(doc_id, 'TRAINING_FAILED', str(e))
        return {"status": "error", "message": str(e)}
