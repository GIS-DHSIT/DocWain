import csv
import hashlib
import json
import logging

from src.utils.logging_utils import get_logger
import math
import os
import re
import subprocess
import time
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
import warnings
warnings.filterwarnings("ignore", message=r".*_target_device.*has been deprecated", category=FutureWarning)
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from urllib.parse import urlparse
try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

from src.api.config import Config
from src.api.context_understanding import ContextUnderstanding
from src.api.pii_masking import mask_document_content
from src.api.dw_document_extractor import DocumentExtractor
from src.api.content_store import delete_extracted_pickle, save_extracted_pickle
from src.api.pipeline_models import ChunkRecord, ExtractedDocument, Section
from src.api.vector_store import QdrantVectorStore, build_collection_name, compute_chunk_id
from src.embedding.pipeline.chunk_integrity import is_valid_chunk_text
from src.embedding.pipeline.embed_pipeline import ChunkPrepStats, prepare_embedding_chunks, normalize_chunk_chain
from src.embedding.pipeline.payload_normalizer import build_qdrant_payload, normalize_chunk_metadata
from src.embedding.pipeline.schema_normalizer import EMBED_PIPELINE_VERSION
from src.kg.ingest import build_graph_payload, get_graph_ingest_queue
from src.embedding.model_loader import encode_with_fallback as model_encode_with_fallback, get_embedding_model
from src.embedding.chunking.section_chunker import SectionChunker, normalize_text
from src.metrics.ai_metrics import get_metrics_store
from src.metrics.telemetry import METRICS_V2_ENABLED, telemetry_store
from src.utils.payload_utils import is_valid_text
from azure.core.exceptions import ResourceNotFoundError

from src.storage.azure_blob_client import (
    classify_blob_error,
    get_container_client,
    get_document_container_client,
    iter_blob_name_candidates,
    normalize_blob_name,
    sanitize_blob_url,
)

# Lazy-loaded globals to avoid heavy initialization during import
docEx = None
_MODEL = None
_MODEL_DEVICE = None
_QDRANT_CLIENT = None
_VECTOR_STORE = None
_HASH_VECTORIZER = None
logger = get_logger(__name__)

'-------------------------------modified by maha/maria-----------------------'
'---------------------------------new function for checking PII status in mongodb--------------------'

class ChunkingDiagnosticError(RuntimeError):
    def __init__(self, message: str, diagnostics: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}

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
                logger.info(
                    f"Subscription {subscription_id}: PII masking is {'ENABLED' if pii_enabled else 'DISABLED'}")
                return bool(pii_enabled)
            else:
                # Persist the default back so subsequent lookups don't repeat the warning
                try:
                    collection.update_one(
                        {"_id": subscription["_id"]},
                        {"$set": {"pii_enabled": True}},
                    )
                    logger.info(
                        "Subscription %s: PII setting not found, persisted default ENABLED",
                        subscription_id,
                    )
                except Exception as persist_exc:
                    logger.warning(
                        "Subscription %s: PII setting not found, defaulting to ENABLED (persist failed: %s)",
                        subscription_id, persist_exc,
                    )
                return True  # Safe default - enable PII masking if not specified
        else:
            logger.info(
                "PII masking defaulted to enabled because subscription not found (subscription_id=%s)",
                subscription_id,
            )
            return True  # Safe default

    except Exception as e:
        logger.error(f"Error fetching PII setting for subscription {subscription_id}: {e}")
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

def create_mongo_client():
    """Create a Mongo client with a graceful fallback when the primary URI is misconfigured."""
    primary_uri = Config.MongoDB.URI
    fallback_uri = getattr(Config.MongoDB, "FALLBACK_URI", None)
    client = None
    try:
        client = MongoClient(primary_uri, serverSelectionTimeoutMS=5000)
        try:
            client.admin.command("ping")
            logger.info(f"Connected to MongoDB primary URI: {primary_uri}")
            return client
        except Exception as ping_exc:
            logger.warning(f"Ping to primary MongoDB URI failed: {ping_exc}")
    except Exception as exc:
        logger.warning(f"Primary MongoDB URI failed ({exc}); attempting fallback")

    if fallback_uri and fallback_uri != primary_uri:
        try:
            fallback_client = MongoClient(fallback_uri, serverSelectionTimeoutMS=5000)
            try:
                fallback_client.admin.command("ping")
                logger.info(f"Connected to MongoDB fallback URI: {fallback_uri}")
            except Exception as fb_ping:
                logger.warning(f"Ping to fallback MongoDB URI failed: {fb_ping}")
            return fallback_client
        except Exception as fb_exc:
            logger.error(f"Fallback MongoDB URI also failed: {fb_exc}")

    if client is not None:
        logger.warning("Using MongoClient without verified connectivity")
        return client
    logger.error("Unable to create MongoClient; falling back to localhost without ping")
    return MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)

mongoClient = create_mongo_client()
db = mongoClient[Config.MongoDB.DB]

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
        logger.debug("CUDA availability check failed")
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
    try:
        if torch and isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception as _check_exc:  # noqa: BLE001
        logger.debug("CUDA OOM type check failed: %s", _check_exc)
    msg = str(exc).lower()
    return (
        "cuda out of memory" in msg
        or "cuda error: out of memory" in msg
        or "cublas_status_alloc_failed" in msg
        or ("cuda error" in msg and "alloc" in msg)
    )

def _clear_gpu_cache() -> None:
    """Best-effort GPU cache clear after CUDA OOM."""
    try:
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        logger.debug("GPU cache clear failed: %s", exc)

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
    # Default to float32 for robustness unless explicitly overridden.
    return torch.float32

def _model_kwargs_for_device(device: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"device_map": None, "low_cpu_mem_usage": False}
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
    logger.info("Loading sentence transformer model: %s (device=%s)", name, device)
    model_kwargs = _model_kwargs_for_device(device)
    try:
        if getattr(Config.Model, "OFFLINE_ONLY", True):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        return SentenceTransformer(name, device=device, model_kwargs=model_kwargs, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        if device != "cpu" and (_is_meta_tensor_error(exc) or _is_cuda_oom(exc)):
            logger.warning(
                "Model load on %s failed (%s); retrying on cpu for stability",
                device,
                exc,
            )
            cpu_kwargs = _model_kwargs_for_device("cpu")
            return SentenceTransformer(name, device="cpu", model_kwargs=cpu_kwargs, local_files_only=True)
        raise

def get_model(*, reload: bool = False, device: Optional[str] = None):
    """Lazy init for sentence transformer model with robust device fallback."""
    global _MODEL, _MODEL_DEVICE
    model, _dim = get_embedding_model(reload=reload, device=device)
    _MODEL = model
    _MODEL_DEVICE = getattr(model, "device", None) or getattr(model, "_target_device", None) or _MODEL_DEVICE
    return model

def _embedding_batch_size(default: int = 32) -> int:
    raw = os.getenv("EMBEDDING_BATCH_SIZE", str(default))
    try:
        return max(1, int(raw))
    except ValueError:
        logger.debug("Invalid EMBEDDING_BATCH_SIZE=%r, using default=%d", raw, default)
        return default

def encode_with_fallback(
    texts: List[str],
    *,
    normalize_embeddings: bool = False,
    convert_to_numpy: bool = True,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
):
    """Encode with recovery from meta-device and CUDA OOM failures."""
    # Let model_loader handle adaptive batch sizing per device
    effective_batch_size = batch_size or _embedding_batch_size()
    return model_encode_with_fallback(
        texts,
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=convert_to_numpy,
        batch_size=effective_batch_size,
        device=device,
    )

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
        logger.info("Decryption successful.")
        return decrypted_text
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        return ""

def fileProcessor(content, file, content_type: str = ""):
    """Processes different types of documents and extracts text or dataframe."""
    extracted_data = {}
    try:
        file_name = file.split('/')[-1]
        extractor = get_doc_extractor()

        # MIME-type fallback: detect format from content_type when extension is
        # missing or generic (e.g. ".bin", no extension at all).
        _ct = (content_type or "").lower()
        _ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
        _KNOWN_EXTS = {"csv", "xlsx", "xls", "json", "pdf", "docx", "doc", "pptx", "ppt", "txt"}
        if _ext not in _KNOWN_EXTS and _ct:
            _MIME_EXT_MAP = {
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
                "application/vnd.ms-excel": "xls",
                "application/pdf": "pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
                "application/msword": "doc",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
                "application/vnd.ms-powerpoint": "ppt",
                "text/csv": "csv",
                "application/json": "json",
                "text/plain": "txt",
            }
            mapped_ext = _MIME_EXT_MAP.get(_ct)
            if mapped_ext:
                logger.info(f"MIME-type fallback: '{_ct}' → .{mapped_ext} for file '{file_name}'")
                file_name = f"{file_name}.{mapped_ext}"

        if content:
            logger.info(f"Extracting File {file_name}")
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
            elif file_name.endswith(".doc"):
                # Legacy .doc files are binary (often UTF-16LE);
                # extract_text_from_txt handles via _smart_decode()
                extracted_data[file_name] = extractor.extract_text_from_txt(content, filename=file_name)
            else:
                # Last resort: try to detect Excel by magic bytes
                if content[:4] == b'PK\x03\x04':
                    # ZIP-based format — could be xlsx/docx/pptx; try Excel first
                    try:
                        sheets = pd.read_excel(BytesIO(content), sheet_name=None)
                        for sheet_name, df in sheets.items():
                            key = f"{file_name}#{sheet_name}"
                            extracted_data[key] = extractor.extract_dataframe(df, sheet_name=sheet_name)
                        logger.info(f"Magic-byte detection: treated '{file_name}' as Excel (xlsx)")
                    except Exception as exc:
                        logger.debug("Excel parse failed for %s, falling back to text: %s", file_name, exc)
                        extracted_data[file_name] = extractor.extract_text_from_txt(content, filename=file_name)
                else:
                    extracted_data[file_name] = extractor.extract_text_from_txt(content, filename=file_name)
        return extracted_data
    except Exception as e:
        logger.error(f"Error processing file {file}: {e}")
        return {}

def read_s3_file(s3, bucket, file_key):
    """Reads a file from S3."""
    try:
        logger.info(f"Reading S3 file: {file_key} from bucket: {bucket}")
        obj = s3.get_object(Bucket=bucket, Key=file_key)
        content = obj["Body"].read()
        logger.info("S3 file read successfully.")
        return content
    except Exception as e:
        logger.error(f"Error reading S3 file {file_key}: {e}")
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
        logger.error(f"Error creating S3 client: {e}")
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
        logger.warning("S3 document fetch failed for %s/%s: %s", bucket_name, object_key, e)
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
            logger.info(f"Training status for document {document_id} updated to {status}")
            return {"status": "success"}
        else:
            logger.warning(f"No document found with ID {document_id}")
            return {"status": "not_found"}
    except Exception as e:
        logger.error(f"Error updating training status for {document_id}: {e}")
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
            from src.api.blob_store import blob_storage_configured

            container_name = (
                os.getenv("DOCWAIN_BLOB_CONTAINER")
                if blob_storage_configured()
                else os.getenv("DOCUMENT_CONTENT_DIR", "document-content")
            )
            update_data["blob_reference"] = {
                "container": container_name,
                "blob_name": Path(pickle_path).name,
            }
        collection = db[Config.MongoDB.DOCUMENTS]
        collection.update_one(filter_criteria, {"$set": update_data})
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Error updating extraction metadata for {document_id}: {exc}")

def update_layout_graph_metadata(
    document_id: str,
    *,
    layout_latest_path: Optional[str],
    layout_versioned_path: Optional[str],
    layout_hash: Optional[str],
) -> None:
    """Persist layout graph pointers without storing the full graph in Mongo."""
    try:
        if ObjectId.is_valid(str(document_id)):
            filter_criteria = {"_id": ObjectId(str(document_id))}
        else:
            filter_criteria = {"_id": str(document_id)}

        update_data = {
            "layout_graph_latest_path": layout_latest_path,
            "layout_graph_versioned_path": layout_versioned_path,
            "layout_graph_hash": layout_hash,
            "layout_graph_updated_at": time.time(),
            "updated_at": time.time(),
        }
        collection = db[Config.MongoDB.DOCUMENTS]
        collection.update_one(filter_criteria, {"$set": update_data})
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Error updating layout graph metadata for {document_id}: {exc}")

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
        logger.error(f"Error updating security screening for {document_id}: {exc}")

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
        logger.error(f"Error updating PII stats for {document_id}: {e}")

def clear_legacy_vetting_metadata() -> None:
    """Remove deprecated vetting metadata from all documents."""
    try:
        collection = db[Config.MongoDB.DOCUMENTS]
        result = collection.update_many({"vettingPoints": {"$exists": True}}, {"$unset": {"vettingPoints": ""}})
        if result.modified_count:
            logger.info("Cleared legacy vetting metadata from %s documents.", result.modified_count)
        else:
            logger.info("Legacy vetting metadata not present; no cleanup needed.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Legacy vetting metadata cleanup skipped: %s", exc)

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
        logger.error(f"Error fetching PII stats for {document_id}: {e}")
        return None

def get_azure_docs(files, *, document_id: Optional[str] = None):
    """Fetch documents from Azure Blob Storage."""
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
            container_client = (
                get_document_container_client()
                if candidate_container == container_name
                else get_container_client(candidate_container)
            )
            name_variants = [blob_name]
            if candidate_container == teams_container and local_prefixed_blob:
                name_variants.insert(0, local_prefixed_blob)
            for name_variant in name_variants:
                for candidate in iter_blob_name_candidates(name_variant):
                    blob_client = container_client.get_blob_client(blob=candidate)
                    blob_url = sanitize_blob_url(getattr(blob_client, "url", ""))
                    if blob_url:
                        logger.info("Downloading blob for document_id=%s url=%s", document_id, blob_url)
                    else:
                        logger.info(
                            "Downloading blob for document_id=%s container=%s blob=%s",
                            document_id,
                            candidate_container,
                            candidate,
                        )
                    try:
                        return blob_client.download_blob().readall()
                    except ResourceNotFoundError as exc:
                        logger.debug("Blob not found at %s/%s, trying next candidate", candidate_container, candidate)
                        last_exc = exc
                        continue
        if last_exc:
            raise last_exc
        raise ResourceNotFoundError(message="Blob name candidates empty", response=None)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        error = classify_blob_error(exc, document_id=document_id, blob_name=blob_name)
        logger.error(
            "Blob download failed document_id=%s blob=%s error_type=%s message=%s request_id=%s",
            document_id,
            blob_name,
            error.__class__.__name__,
            exc,
            error.request_id,
        )
        raise error from exc

'-------------------------------modified by maha/maria-----------------------'
'---------------Added a PII control per subscription in connectData function--------------------'

def connectData(documentConnection):
    """
     FINAL FIX: Ensures each document processes ONLY its exact file

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
            logger.error(f"Subscription resolution failed for document {docId}: {exc}")
            update_training_status(docId, 'TRAINING_FAILED', 'subscription_id missing')
            continue

        # Check PII setting for this subscription
        pii_masking_enabled = get_subscription_pii_setting(subscriptionId)
        logger.info(f"Document {docId} (Subscription {subscriptionId}): PII masking = {pii_masking_enabled}")

        allowed_statuses = {'UNDER_REVIEW', 'TRAINING_FAILED'}

        if docData.get('status') in allowed_statuses:
            try:
                logger.info(f"=" * 80)
                logger.info(f"Processing document {docId}: {docData.get('name', 'Unknown')}")
                logger.info(f"=" * 80)
                if telemetry:
                    try:
                        telemetry.record_metadata_quality(docId, docData, expected_fields=list(docData.keys()))
                    except Exception as exc:
                        logger.debug("Metadata quality recording failed for %s: %s", docId, exc)
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
                        logger.error(f"Failed to create S3 client for document {docId}")
                        update_training_status(docId, 'TRAINING_FAILED', 'Failed to create S3 client')
                        continue

                    objs = s3.list_objects_v2(Bucket=bkName)
                    file = [obj['Key'] for obj in objs.get("Contents", []) if obj['Key'] == docData['name']]

                    if not file:
                        logger.error(f"File {docData['name']} not found in S3 bucket {bkName}")
                        update_training_status(docId, 'TRAINING_FAILED', 'File not found in S3')
                        continue

                    docContent = read_s3_file(s3, bkName, file[0])
                    if docContent is None:
                        logger.error(f"Failed to read S3 file for document {docId}")
                        update_training_status(docId, 'TRAINING_FAILED', 'Failed to read S3 file')
                        continue

                    extractedDoc = fileProcessor(docContent, file[0])
                    if not extractedDoc:
                        logger.error(f"Failed to extract content from document {docId}")
                        update_training_status(docId, 'TRAINING_FAILED', 'Content extraction failed')
                        continue

                    all_extracted_docs.update(extractedDoc)

                elif docData['type'] == 'LOCAL':
                    # ============================================
                    #  CRITICAL FIX: EXACT filename matching
                    # ============================================

                    doc_name = docData.get('name', '')

                    if not doc_name:
                        logger.error(f"No filename found for document {docId}")
                        update_training_status(docId, 'TRAINING_FAILED', 'No filename specified')
                        continue

                    # Get all files from connector
                    all_connector_files = connData.get('locations', [])

                    logger.info(f"Looking for EXACT match: '{doc_name}'")
                    logger.info(f"Connector has {len(all_connector_files)} files")

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
                            logger.info(f" EXACT MATCH: {file_path} matches {doc_name}")
                            break  # Stop after first exact match

                    # L If no exact match, DO NOT fallback to partial matching
                    if not matching_files:
                        logger.error(f"L NO EXACT MATCH for document {docId} (name: '{doc_name}')")
                        logger.error(f"Available files in connector:")
                        for f in all_connector_files:
                            file_only = f.split('/')[-1] if '/' in f else f
                            logger.error(f"  - {file_only} (full path: {f})")
                        update_training_status(docId, 'TRAINING_FAILED', f'Exact file match not found: {doc_name}')
                        continue

                    #  Should only have ONE match due to break statement
                    if len(matching_files) > 1:
                        logger.error(
                            f"L CRITICAL: Multiple exact matches for {doc_name}: {matching_files}"
                        )
                        logger.error("This should not happen with exact matching!")
                        update_training_status(docId, 'TRAINING_FAILED', 'Multiple file matches')
                        continue

                    file_path = matching_files[0]
                    logger.info(f" Document {docId} will process ONLY: {file_path}")

                    # Process ONLY this document's specific file
                    try:
                        # Extract the blob name after 'az://' prefix while preserving nested paths
                        file_key = normalize_blob_name(
                            file_path, container_name=Config.AzureBlob.DOCUMENT_CONTAINER_NAME
                        )
                        logger.info(f"Reading file: {file_key} for document {docId}")

                        docContent = get_azure_docs(file_key, document_id=docId)
                        if docContent is None:
                            logger.error(f"Failed to read Azure file {file_key} for document {docId}")
                            update_training_status(docId, 'TRAINING_FAILED', f'Failed to read file {file_key}')
                            continue

                        extractedDoc = fileProcessor(docContent, file_path)
                        if not extractedDoc:
                            logger.error(f"Failed to extract content from file {file_key}")
                            update_training_status(docId, 'TRAINING_FAILED', 'Content extraction failed')
                            continue

                        all_extracted_docs.update(extractedDoc)

                        #  VERIFICATION: Should only have ONE file extracted
                        if len(all_extracted_docs) != 1:
                            logger.warning(
                                f"� Expected 1 extracted file for {docId}, got {len(all_extracted_docs)}: "
                                f"{list(all_extracted_docs.keys())}"
                            )
                        else:
                            logger.info(f" Successfully extracted 1 file from {file_key}")

                    except Exception as file_error:
                        logger.error(f"Error processing file {file_path}: {file_error}")
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
                        logger.error(f"Failed to persist extracted pickle for {docId}: {exc}")
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
                            logger.error(f"Security screening failed for document {docId}; blocking training")
                            update_training_status(docId, 'TRAINING_BLOCKED_SECURITY', 'Security screening failed')
                            continue
                    except Exception as exc:
                        logger.error(f"Security screening failed for {docId}: {exc}")
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
                        logger.info(f"PII masking disabled for subscription {subscriptionId}")
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

                    logger.info(
                        f" Stored document {docId} with {len(masked_docs)} file(s) "
                        f"(PII masked: {pii_count})"
                    )
                else:
                    logger.error(f"No documents extracted for {docId}")
                    update_training_status(docId, 'TRAINING_FAILED', 'No content extracted')

            except Exception as e:
                logger.error(f"Error processing document {docId} ({docData.get('name', 'Unknown')}): {e}")
                update_training_status(docId, 'TRAINING_FAILED', str(e))

        elif docData['status'] == 'DELETED':
            profileData = str(docData['profile'])
            delete_embeddings(subscriptionId, profileData, docId)

    logger.info(f"=" * 80)
    logger.info(f" connectData completed: {len(dataDict)} documents processed")
    logger.info(f"=" * 80)

    return dataDict

def collectionConnect(name):
    """Fetches documents from a MongoDB collection."""
    logger.info(f"Fetching connection details for collection: {name}")
    try:
        collection = db[name]
        try:
            # Count documents to make emptiness explicit in the logs
            count = collection.count_documents({})
            logger.info(f"Collection '{name}' document count: {count}")
        except Exception as count_exc:
            logger.warning(f"Unable to count documents for collection '{name}': {count_exc}")
        return collection.find()
    except Exception as e:
        logger.error(f"Error connecting to collection {name}: {e}")
        # return an empty iterator to keep calling code behavior predictable
        return []

def extract_document_info():
    """Retrieves connector details from MongoDB."""
    try:
        logger.info(
            f"Extracting document info from DB: {Config.MongoDB.DB}, collections: {Config.MongoDB.DOCUMENTS}, {Config.MongoDB.CONNECTOR}")
        try:
            existing = db.list_collection_names()
            logger.info(f"Existing collections in DB '{Config.MongoDB.DB}': {existing}")
        except Exception as lc_exc:
            logger.warning(f"Could not list collections: {lc_exc}")

        docs = collectionConnect(Config.MongoDB.DOCUMENTS)
        Docs = {}
        # If docs is a cursor/iterable, iterate, otherwise it's probably an empty list
        for doc in docs:
            try:
                refConnector = doc.get('_id').__str__()
            except Exception as exc:
                logger.debug("Document _id extraction fallback: %s", exc)
                refConnector = str(doc.get('_id', 'unknown'))
            Docs[refConnector] = doc
        logger.info(f"Found {len(Docs)} document definitions in collection '{Config.MongoDB.DOCUMENTS}'")

        connInfo = {}
        connectors = collectionConnect(Config.MongoDB.CONNECTOR)
        for conn in connectors:
            try:
                connId = conn.get('_id').__str__()
            except Exception as exc:
                logger.debug("Connector _id extraction fallback: %s", exc)
                connId = str(conn.get('_id', 'unknown'))
            connInfo[connId] = conn
        logger.info(f"Found {len(connInfo)} connector definitions in collection '{Config.MongoDB.CONNECTOR}'")

        connList = list(connInfo.keys())
        docInfo = {}
        for docId, docData in Docs.items():
            # docData expected to have a 'connector' field referencing a connector id
            try:
                connRef = docData.get('connector').__str__()
            except Exception as exc:
                logger.debug("Connector ref extraction fallback for doc %s: %s", docId, exc)
                connRef = str(docData.get('connector', ''))
            if connRef in connList:
                docInfo[docId] = {'dataDict': docData, 'connDict': connInfo[connRef]}
            else:
                logger.info(f"Document {docId} references connector {connRef} which is not in connectors list")

        logger.info(f"Extracted {len(docInfo)} documents with valid connectors")
        return docInfo

    except Exception as e:
        logger.error(f"Error fetching connection details: {e}")
        return {}

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
        logger.info(
            f"[DELETE_EMBEDDINGS] Deleting embeddings for document_id={document_id}, "
            f"subscription_id={subscription_id}, profile_id={profile_id}"
        )

        if not profile_id:
            raise ValueError("profile_id is required for deleting embeddings")

        store = get_vector_store()
        result = store.delete_document(subscription_id, profile_id, document_id)
        return result

    except Exception as e:
        error_msg = str(e)
        if "doesn't exist" in error_msg or "not found" in error_msg.lower() or "Not Found" in error_msg:
            logger.debug("[DELETE_EMBEDDINGS] Collection not found (no embeddings to delete) for doc=%s", document_id)
            return {"status": "ok", "message": "no embeddings to delete", "document_id": document_id}
        logger.error("[DELETE_EMBEDDINGS] Failed for doc=%s: %s", document_id, error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "document_id": document_id,
        }

def ensure_qdrant_collection(collection_name: str, vector_size: int) -> None:
    """Ensure Qdrant collection exists with multi-vector schema and payload indexes."""
    try:
        store = get_vector_store()
        store.ensure_collection(collection_name, vector_size)
    except Exception as e:
        logger.error(f"Error ensuring collection in Qdrant: {e}")
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
            logger.error(f"Error: 'embeddings' key missing or None for document {doctag}!")
            raise ValueError("Embeddings data is invalid")

        raw_vectors = embeddings["embeddings"]

        # Fallback: if embeddings are actually plain text strings, re-embed them
        if isinstance(raw_vectors, (list, tuple)) and raw_vectors and all(isinstance(v, str) for v in raw_vectors):
            logger.warning(
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

        doc_domain = embeddings.get("doc_domain")
        full_text_seed = (
            embeddings.get("full_text")
            or doc_metadata.get("full_text")
            or " ".join([t for t in texts if t])
        )
        try:
            from src.utils.language import detect_language

            if not languages:
                lang, confidence = detect_language(full_text_seed or "")
                if lang and lang != "unknown":
                    languages = [lang]
                doc_metadata["detected_language"] = lang
                doc_metadata["language_confidence"] = confidence
        except Exception as exc:  # noqa: BLE001
            logger.debug("Language detection failed: %s", exc)
        if languages:
            doc_metadata["languages"] = languages
        if doc_metadata.get("detected_language"):
            embeddings["detected_language"] = doc_metadata.get("detected_language")
        if doc_metadata.get("language_confidence") is not None:
            embeddings["language_confidence"] = doc_metadata.get("language_confidence")
        if languages:
            embeddings["languages"] = languages
        doc_version_hash = (
            embeddings.get("doc_version_hash")
            or doc_metadata.get("doc_version_hash")
            or hashlib.sha1((full_text_seed or "").encode("utf-8")).hexdigest()[:12]
        )
        embeddings["doc_version_hash"] = doc_version_hash
        section_intel_result = None
        section_intel_payload = embeddings.get("section_intelligence")
        if isinstance(section_intel_payload, dict) and section_intel_payload.get("sections"):
            section_intel_result = section_intel_payload
            doc_domain = section_intel_payload.get("doc_domain") or doc_domain
            embeddings["doc_domain"] = doc_domain
        elif texts:
            try:
                from src.intelligence.section_intelligence_builder import SectionIntelligenceBuilder

                telemetry = telemetry_store() if METRICS_V2_ENABLED else None
                section_start = time.time()
                full_text = embeddings.get("full_text") or " ".join([t for t in texts if t])
                builder = SectionIntelligenceBuilder()
                section_intel_result = builder.build(
                    document_id=str(doctag),
                    document_text=full_text,
                    chunk_texts=list(texts),
                    chunk_metadata=chunk_metadata,
                    metadata={
                        "doc_type": doc_type or document_type,
                        "source_name": filename or source_filename,
                    },
                )
                if telemetry:
                    telemetry.observe("section_build_time_ms", (time.time() - section_start) * 1000)
                doc_domain = section_intel_result.doc_domain or doc_domain
                embeddings["section_intelligence"] = section_intel_result.to_dict()
                embeddings["doc_domain"] = doc_domain
            except Exception as exc:  # noqa: BLE001
                logger.debug("Section intelligence build skipped for %s: %s", doctag, exc)
                if not doc_domain:
                    try:
                        from src.intelligence.domain_indexer import infer_domain

                        sample_text = " ".join([t for t in texts[:5] if t])
                        doc_domain = infer_domain(
                            sample_text,
                            doc_type=doc_type or document_type,
                            source_name=filename or source_filename,
                        )
                    except Exception as exc:
                        logger.debug("Domain inference (section intel) failed: %s", exc)
                        doc_domain = None

        if not doc_domain or str(doc_domain).strip().lower() in {"unknown", "generic"}:
            try:
                from src.intelligence.domain_indexer import infer_domain

                doc_domain = infer_domain(
                    full_text_seed or " ".join([t for t in texts if t]),
                    doc_type=doc_type or document_type,
                    source_name=filename or source_filename,
                )
            except Exception as exc:
                logger.debug("Domain inference fallback failed: %s", exc)
                doc_domain = doc_domain or "unknown"

        # Final safety net — use content classifier if still unknown
        if not doc_domain or str(doc_domain).strip().lower() in {"unknown", ""}:
            from src.embedding.pipeline.content_classifier import classify_doc_domain

            doc_domain = classify_doc_domain(
                full_text_seed or " ".join([t for t in texts if t]),
                filename or source_filename,
                doc_type or document_type or "",
            )
        embeddings["doc_domain"] = doc_domain

        if not texts:
            raise ValueError(f"No texts found for document {doctag}")

        if len(texts) != len(normalized_vectors):
            raise ValueError(
                f"Embeddings/text length mismatch for document {doctag}: "
                f"texts={len(texts)} embeddings={len(normalized_vectors)}"
            )

        max_len = len(texts)
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

        # Delete old points for this document before upserting to prevent duplicates
        old_points_deleted = False
        try:
            store = get_vector_store()
            store.delete_document(subscription_id, profile_id, doctag)
            old_points_deleted = True
            logger.info(
                "Cleaned old embeddings for document_id=%s before re-embedding",
                doctag,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Old embedding cleanup failed for %s: %s", doctag, exc)

        # Apply embedding enhancement for better retrieval
        dropped_dedup = 0
        try:
            from src.embedding.pipeline_enhancement import enhance_chunks_for_embedding

            enhancement_result = enhance_chunks_for_embedding(
                texts=list(texts),
                chunk_metadata=list(chunk_metadata) if chunk_metadata else [],
                document_metadata={
                    "document_id": str(doctag),
                    "document_type": doc_type or document_type,
                    "document_domain": doc_domain,
                },
                domain=doc_domain or "generic",
            )

            # Apply deduplicated texts and metadata from enhancement
            dropped_dedup = enhancement_result.original_count - enhancement_result.deduplicated_count
            if enhancement_result.deduplicated_count < enhancement_result.original_count:
                deduped_texts = enhancement_result.enhanced_texts
                deduped_meta = enhancement_result.enhanced_metadata
                # Re-embed since the count changed after deduplication
                deduped_vectors, _ = normalize_embedding_matrix(
                    encode_with_fallback(
                        deduped_texts,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    )
                )
                deduped_sparse = build_sparse_vectors(deduped_texts)
                # Atomic swap: only replace if re-encoding succeeded
                texts = deduped_texts
                chunk_metadata = deduped_meta
                normalized_vectors = deduped_vectors
                sparse_vectors = deduped_sparse
                max_len = len(texts)
            else:
                # No dedup needed — just update metadata in-place
                if enhancement_result.enhanced_metadata:
                    for idx, enhanced_meta in enumerate(enhancement_result.enhanced_metadata):
                        if idx < len(chunk_metadata):
                            chunk_metadata[idx].update(enhanced_meta)
                        else:
                            chunk_metadata.append(enhanced_meta)

            logger.info(
                f"Embedding enhancement: {enhancement_result.original_count} -> "
                f"{enhancement_result.deduplicated_count} chunks, "
                f"avg quality: {enhancement_result.average_quality_score:.2f}"
            )
        except Exception as enhance_exc:  # noqa: BLE001
            logger.warning("Embedding enhancement failed for %s: %s", doctag, enhance_exc)

        records: List[ChunkRecord] = []
        invalid_samples: List[Dict[str, Any]] = []
        chunk_errors: int = 0
        min_chars = int(getattr(Config.Retrieval, "MIN_CHARS", 50))
        min_tokens = int(getattr(Config.Retrieval, "MIN_TOKENS", 10))
        for idx in range(max_len):
            try:
                vector = normalized_vectors[idx]
                text = texts[idx]
                if not vector:
                    continue
                if not is_valid_chunk_text(text, min_chars=min_chars, min_tokens=min_tokens):
                    chunk_meta = chunk_metadata[idx] if idx < len(chunk_metadata) else {}
                    invalid_samples.append(
                        {
                            "chunk_id": (chunk_meta or {}).get("chunk_id"),
                            "length": len(text or ""),
                            "sample": _sanitize_text_sample(text),
                        }
                    )
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
                section_path = chunk_meta.get("section_path") or section_val or "Untitled Section"
                section_id = chunk_meta.get("section_id")
                section_kind = chunk_meta.get("section_kind") or "misc"
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

                embedding_text = chunk_meta.get("embedding_text") or chunk_meta.get("text_clean") or text
                content_text = chunk_meta.get("content") or chunk_meta.get("text_raw") or chunk_meta.get("text_clean") or text
                # Single source of truth: pass flat fields to build_qdrant_payload()
                raw_payload = {
                    **chunk_meta,
                    "subscription_id": str(subscription_id),
                    "profile_id": str(profile_id),
                    "document_id": str(doctag),
                    "content": content_text,
                    "embedding_text": embedding_text,
                    "source_name": filename or source_filename,
                    "doc_domain": doc_domain,
                    "section_id": section_id,
                    "section_title": section_val,
                    "section_path": section_path,
                    "section_kind": section_kind,
                    "page": page_val,
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "chunk_count": max_len,
                    "chunk_kind": chunk_kind,
                    "hash": chunk_meta.get("chunk_hash") or chunk_hash,
                    "doc_version_hash": doc_version_hash,
                    "embed_pipeline_version": EMBED_PIPELINE_VERSION,
                    "doc_summary": (doc_metadata.get("document_summary") or "")[:500] or None,
                    "doc_key_entities": doc_metadata.get("key_entities") or None,
                    "doc_intent_tags": doc_metadata.get("intent_tags") or None,
                    "quality_grade": doc_metadata.get("quality_grade") or None,
                    "domain_signals": doc_metadata.get("domain_signals") or None,
                    "entity_count": doc_metadata.get("entity_count") or 0,
                    "section_role": doc_metadata.get("section_roles", {}).get(section_val) if section_val else None,
                    "temporal_span": doc_metadata.get("temporal_span") or None,
                }

                payload = build_qdrant_payload(raw_payload)

                records.append(
                    ChunkRecord(
                        chunk_id=str(chunk_id),
                        dense_vector=[float(x) for x in vector],
                        sparse_vector=sparse_vector,
                        payload=payload,
                    )
                )
            except Exception as chunk_exc:  # noqa: BLE001
                chunk_errors += 1
                logger.warning(
                    "Chunk %d/%d failed for doc %s: %s",
                    idx + 1, max_len, doctag, chunk_exc,
                )
                if _is_cuda_oom(chunk_exc):
                    _clear_gpu_cache()
                continue  # Skip bad chunk, don't fail entire document

        if not records:
            raise ValueError(f"No valid embedding records generated for document {doctag}")

        try:
            saved = get_vector_store().upsert_records(collection_name, records, batch_size=batch_size)
        except Exception as upsert_exc:
            if old_points_deleted:
                logger.error(
                    "CRITICAL: Old embeddings deleted but re-upsert failed for %s — document has ZERO vectors: %s",
                    doctag, upsert_exc,
                )
            raise
        logger.info(
            f"Saved {saved} embeddings for document {doctag} in collection {collection_name} (profile={profile_id})"
        )
        if invalid_samples:
            logger.warning(
                "Skipped %s invalid chunks during embedding for %s (min_chars=%s min_tokens=%s)",
                len(invalid_samples),
                doctag,
                min_chars,
                min_tokens,
            )
        if chunk_errors:
            logger.warning(
                "Per-chunk errors during record building for %s: %d/%d chunks failed",
                doctag, chunk_errors, max_len,
            )

        try:
            from src.api.dw_newron import get_redis_client
        except Exception as exc:  # noqa: BLE001
            logger.debug("Redis client import unavailable: %s", exc)
            get_redis_client = None

        redis_client = None
        if get_redis_client:
            try:
                redis_client = get_redis_client()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Redis client initialization failed: %s", exc)
                redis_client = None

        graph_payload = build_graph_payload(
            embeddings_payload=embeddings,
            subscription_id=subscription_id,
            profile_id=profile_id,
            document_id=doctag,
            doc_name=filename or source_filename,
            doc_metadata=doc_metadata,
        )
        if graph_payload:
            queue = get_graph_ingest_queue(redis_client)
            queue.enqueue(graph_payload)

        try:
            from src.intelligence.facts_store import FactsStore
            from src.intelligence.kg_updater import KGUpdater
            from src.intelligence.domain_indexer import DomainIndexer
            from src.intelligence.redis_intel_cache import RedisIntelCache

            facts_store = FactsStore(redis_client=redis_client, db=db)
            if section_intel_result:
                if isinstance(section_intel_result, dict):
                    sections_payload = section_intel_result.get("sections") or []
                    facts_payload = section_intel_result.get("section_facts") or []
                    summaries_payload = section_intel_result.get("section_summaries") or {}
                else:
                    sections_payload = [sec.__dict__ for sec in section_intel_result.sections]
                    facts_payload = section_intel_result.section_facts
                    summaries_payload = section_intel_result.section_summaries
                facts_store.persist_document_sections(
                    subscription_id=str(subscription_id),
                    profile_id=str(profile_id),
                    document_id=str(doctag),
                    source_name=filename or source_filename,
                    doc_domain=doc_domain or "generic",
                    doc_version_hash=doc_version_hash,
                    sections=sections_payload,
                    section_facts=facts_payload,
                    section_summaries=summaries_payload,
                )
                kg_updater = KGUpdater(redis_client=redis_client)
                kg_updater.update(
                    subscription_id=str(subscription_id),
                    profile_id=str(profile_id),
                    document_id=str(doctag),
                    source_name=filename or source_filename,
                    doc_domain=doc_domain or "generic",
                    sections=sections_payload,
                    chunk_metadata=chunk_metadata,
                    section_facts=facts_payload,
                )
                if redis_client:
                    cache = RedisIntelCache(redis_client)
                    DomainIndexer(redis_cache=cache).update_entities_only(
                        subscription_id=str(subscription_id),
                        profile_id=str(profile_id),
                        document_id=str(doctag),
                        chunk_texts=list(texts),
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("DWX section intelligence/KG update skipped: %s", exc)

        try:
            from src.embed.embed_pipeline import update_profile_indexes_from_embeddings
            from src.kg.kg_store import KGStore

            if redis_client:
                update_profile_indexes_from_embeddings(
                    embeddings_payload=embeddings,
                    subscription_id=str(subscription_id),
                    profile_id=str(profile_id),
                    document_id=str(doctag),
                    file_name=filename or source_filename,
                    redis_client=redis_client,
                    kg_store=KGStore(),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("DocWain profile index update skipped: %s", exc)

        try:
            if doc_domain:
                collection = db[Config.MongoDB.DOCUMENTS]
                filter_criteria = {"_id": ObjectId(str(doctag))} if ObjectId.is_valid(str(doctag)) else {"_id": str(doctag)}
                update_fields = {"doc_domain": doc_domain, "updated_at": time.time()}
                if languages:
                    update_fields["languages"] = languages
                if doc_metadata.get("detected_language"):
                    update_fields["detected_language"] = doc_metadata.get("detected_language")
                if doc_metadata.get("language_confidence") is not None:
                    update_fields["language_confidence"] = doc_metadata.get("language_confidence")
                collection.update_one(filter_criteria, {"$set": update_fields})
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to update doc_domain metadata for %s: %s", doctag, exc)

        return {
            "status": "success",
            "points_saved": saved,
            "dropped_invalid": len(invalid_samples) + chunk_errors,
            "dropped_dedup": dropped_dedup,
            "invalid_samples": invalid_samples[:5],
            "chunk_errors": chunk_errors,
        }

    except Exception as e:
        logger.error(f"Error saving embeddings to Qdrant for document {doctag}, file {source_filename}: {e}")
        raise

# from enhanced_retrieval import chunk_text_for_embedding
# def train_on_document(text, subscription_id, profile_tag, doc_tag, doc_name):
#     """Trains and stores embeddings with enhanced chunking."""
#     try:
#         logger.info(f"Starting training for {doc_name}")
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
#             logger.info(f"Created {len(chunks)} enhanced chunks for {doc_name}")
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
#         logger.error(f"Training error for {doc_name}: {e}")
#         raise

# CRITICAL FIX for train_on_document() in dataHandler.py

# Replace your existing train_on_document function with this:

from src.api.enhanced_retrieval import chunk_text_for_embedding

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
        logger.debug("Document metadata lookup failed for %s: %s", doc_tag, exc)
        record = {}

    doc_type = (
        record.get("doc_type")
        or record.get("document_type")
        or record.get("type")
        or doc_type_hint
        or ""
    )
    languages = _coerce_list(record.get("languages") or record.get("language"))
    products_name = record.get("products_name") or record.get("product_name") or record.get("product")
    document_type = record.get("document_type") or record.get("doc_type") or doc_type_hint
    description = record.get("description") or record.get("summary") or ""

    filename = _safe_basename(record.get("name") or doc_name)
    profile_name = record.get("profile_name") or record.get("profileName")

    # Document understanding fields (populated by extraction_service)
    document_summary = str(record.get("document_summary") or "")[:500] or None
    key_entities = record.get("key_entities") or []
    intent_tags = record.get("doc_intent_tags") or []

    # Deep analysis fields (populated by deep_analyzer)
    quality_grade = record.get("quality_grade") or None
    quality_score = record.get("quality_score")
    domain_signals = record.get("domain_signals") or {}
    entity_count = len(record.get("entity_mentions") or [])
    section_roles = record.get("section_roles") or {}
    temporal_span = (record.get("chronological_span") or [None])[0] if record.get("chronological_span") else None

    return {
        "filename": filename or _safe_basename(doc_name),
        "doc_type": str(doc_type) if doc_type else None,
        "languages": languages,
        "products_name": str(products_name) if products_name else None,
        "document_type": str(document_type) if document_type else None,
        "description": str(description) if description else None,
        "profile_name": str(profile_name) if profile_name else None,
        "document_summary": document_summary,
        "key_entities": key_entities,
        "intent_tags": intent_tags,
        "quality_grade": quality_grade,
        "quality_score": quality_score,
        "domain_signals": domain_signals,
        "entity_count": entity_count,
        "section_roles": section_roles,
        "temporal_span": temporal_span,
    }

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

def _sanitize_text_sample(text: Optional[str], limit: int = 80) -> str:
    sample = re.sub(r"\s+", " ", (text or "").strip())
    if len(sample) > limit:
        sample = sample[:limit].rstrip() + "..."
    return sample

def _filter_invalid_chunk_texts(
    chunks: List[str],
    metadata: List[Dict[str, Any]],
    *,
    min_chars: int,
    min_tokens: int,
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any], List[int]]:
    valid_chunks: List[str] = []
    valid_meta: List[Dict[str, Any]] = []
    valid_indices: List[int] = []
    invalid_samples: List[Dict[str, Any]] = []
    invalid_lengths: List[int] = []

    for idx, (text, meta) in enumerate(zip(chunks, metadata)):
        if is_valid_chunk_text(text, min_chars=min_chars, min_tokens=min_tokens):
            valid_chunks.append(text)
            valid_meta.append(meta)
            valid_indices.append(idx)
        else:
            invalid_lengths.append(len(text or ""))
            invalid_samples.append(
                {
                    "chunk_id": (meta or {}).get("chunk_id"),
                    "length": len(text or ""),
                    "sample": _sanitize_text_sample(text),
                }
            )

    length_stats = {}
    if invalid_lengths:
        length_stats = {
            "min": min(invalid_lengths),
            "max": max(invalid_lengths),
            "avg": float(sum(invalid_lengths) / max(1, len(invalid_lengths))),
        }

    diagnostics = {
        "min_chars": int(min_chars),
        "min_tokens": int(min_tokens),
        "total_chunks": len(chunks),
        "valid_chunks": len(valid_chunks),
        "invalid_chunks": len(invalid_samples),
        "invalid_length_stats": length_stats,
        "invalid_samples": invalid_samples[:5],
    }

    return valid_chunks, valid_meta, diagnostics, valid_indices

def _mark_chunking_failed(doc_tag: str, doc_name: str, diagnostics: Dict[str, Any]) -> None:
    try:
        from src.api.document_status import update_stage, update_document_fields

        update_stage(
            doc_tag,
            "chunking",
            {
                "status": "EXTRACTION_OR_CHUNKING_FAILED",
                "completed_at": time.time(),
                "diagnostics": diagnostics,
                "doc_name": doc_name,
            },
        )
        update_document_fields(
            doc_tag,
            {
                "chunking_status": "EXTRACTION_OR_CHUNKING_FAILED",
                "chunking_failed_at": time.time(),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to record chunking diagnostics for %s: %s", doc_tag, exc)

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
    except Exception as exc:
        logger.debug("CSV table row parsing failed, returning %d partial rows: %s", len(rows), exc)
        return rows
    return rows

def _build_chunk_metadata_from_section_chunks(
    section_chunks: List[Any],
    *,
    doc_tag: str,
    doc_type: Optional[str],
    doc_ocr_confidence: Optional[float],
    chunking_mode: str = "section_aware",
) -> List[Dict[str, Any]]:
    from src.embedding.pipeline.content_classifier import classify_section_kind

    metadata: List[Dict[str, Any]] = []
    for chunk in section_chunks:
        section_title = (getattr(chunk, "section_title", "") or "Untitled Section").strip() or "Untitled Section"
        section_path = (getattr(chunk, "section_path", "") or section_title).strip() or section_title
        page_start = getattr(chunk, "page_start", None)
        page_end = getattr(chunk, "page_end", None)
        chunk_index = int(getattr(chunk, "chunk_index", len(metadata)))
        sentence_complete = bool(getattr(chunk, "sentence_complete", False))
        chunk_text = getattr(chunk, "text", "") or ""
        section_kind = classify_section_kind(chunk_text, section_title)
        metadata.append(
            {
                "document_id": doc_tag,
                "section_title": section_title,
                "section_path": section_path,
                "section_kind": section_kind,
                "page_start": page_start,
                "page_end": page_end,
                "page_number": page_start,
                "chunk_index": chunk_index,
                "chunk_type": "text",
                "doc_type": doc_type,
                "ocr_confidence": doc_ocr_confidence,
                "sentence_complete": sentence_complete,
                "chunking_mode": chunking_mode,
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
    chunking_mode: str = "section_aware",
) -> Tuple[List[str], List[Dict[str, Any]], Optional[float]]:
    chunker = SectionChunker()
    section_chunks = chunker.chunk_document(content, doc_internal_id=doc_tag, source_filename=doc_name)
    base_metadata = _build_chunk_metadata_from_section_chunks(
        section_chunks,
        doc_tag=doc_tag,
        doc_type=doc_type,
        doc_ocr_confidence=doc_ocr_confidence,
        chunking_mode=chunking_mode,
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

def _fallback_chunk_text_sliding_window(
    text: str,
    *,
    min_required: int,
    min_chars: int,
    min_tokens: int,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    if not text or not str(text).strip():
        return []
    normalized = normalize_text(text)
    tokens = list(re.finditer(r"\S+", normalized))
    if not tokens:
        return []

    total_tokens = len(tokens)
    total_chars = len(normalized.strip())
    if total_tokens < min_required * min_tokens or total_chars < min_required * min_chars:
        return [normalized.strip()]

    chunk_size_tokens = max(min_tokens, int(chunk_size_tokens or 0))
    if total_tokens < chunk_size_tokens:
        chunk_size_tokens = max(min_tokens, int(math.ceil(total_tokens / max(1, min_required))))

    overlap_tokens = max(0, int(overlap_tokens or 0))
    if overlap_tokens >= chunk_size_tokens:
        overlap_tokens = max(0, chunk_size_tokens // 4)
    step = max(1, chunk_size_tokens - overlap_tokens)

    def _slice_chunks(size_tokens: int) -> List[str]:
        local_chunks: List[str] = []
        start = 0
        while start < total_tokens:
            end = min(total_tokens, start + size_tokens)
            start_char = tokens[start].start()
            end_char = tokens[end - 1].end()
            chunk_text = normalized[start_char:end_char].strip()
            if chunk_text:
                local_chunks.append(chunk_text)
            if end >= total_tokens:
                break
            start += step
        return local_chunks

    chunks = _slice_chunks(chunk_size_tokens)
    if len(chunks) < min_required:
        target_size = max(min_tokens, int(math.ceil(total_tokens / max(1, min_required))))
        if target_size < chunk_size_tokens:
            chunk_size_tokens = target_size
            overlap_tokens = min(overlap_tokens, max(0, chunk_size_tokens // 4))
            step = max(1, chunk_size_tokens - overlap_tokens)
            chunks = _slice_chunks(chunk_size_tokens)

    return chunks

def _build_fallback_metadata(
    chunks: List[str],
    *,
    doc_tag: str,
    doc_type: Optional[str],
    doc_ocr_confidence: Optional[float],
) -> List[Dict[str, Any]]:
    metadata: List[Dict[str, Any]] = []
    for idx, text in enumerate(chunks):
        section_path = f"Full Document > Chunk {idx + 1}"
        metadata.append(
            {
                "document_id": doc_tag,
                "section_title": "Full Document",
                "section_path": section_path,
                "page_start": None,
                "page_end": None,
                "page_number": None,
                "chunk_index": idx,
                "chunk_type": "text",
                "chunk_kind": "section_text",
                "doc_type": doc_type,
                "ocr_confidence": doc_ocr_confidence,
                "sentence_complete": str(text).strip().endswith((".", "?", "!")),
                "chunking_mode": "sliding_window_fallback",
            }
        )
    return metadata

def _fallback_chunks_for_full_text(
    full_text: str,
    *,
    doc_tag: str,
    doc_name: str,
    subscription_id: str,
    profile_id: str,
    doc_type: Optional[str],
    doc_ocr_confidence: Optional[float],
    min_required: int,
    min_chars: int,
    min_tokens: int,
) -> Tuple[List[str], List[Dict[str, Any]], Optional[ChunkPrepStats], Dict[str, Any], List[int]]:
    fallback_chunks = _fallback_chunk_text_sliding_window(
        full_text,
        min_required=min_required,
        min_chars=min_chars,
        min_tokens=min_tokens,
        chunk_size_tokens=int(getattr(Config.Retrieval, "FALLBACK_CHUNK_SIZE", 600)),
        overlap_tokens=int(getattr(Config.Retrieval, "FALLBACK_OVERLAP", 80)),
    )
    if not fallback_chunks:
        return [], [], None, {}, []

    fallback_meta = _build_fallback_metadata(
        fallback_chunks,
        doc_tag=doc_tag,
        doc_type=doc_type,
        doc_ocr_confidence=doc_ocr_confidence,
    )
    fallback_chunks, fallback_meta, prep_stats, _rescued = prepare_embedding_chunks(
        fallback_chunks,
        fallback_meta,
        subscription_id=subscription_id,
        profile_id=profile_id,
        document_id=doc_tag,
        doc_name=doc_name,
        quality_filter=_apply_chunk_quality_filter,
    )
    fallback_chunks, fallback_meta, validity_diag, valid_indices = _filter_invalid_chunk_texts(
        fallback_chunks,
        fallback_meta,
        min_chars=min_chars,
        min_tokens=min_tokens,
    )
    return fallback_chunks, fallback_meta, prep_stats, validity_diag, valid_indices

def _extract_text_from_item(item: Any) -> str:
    """Extract clean text from a texts list item that may be a string or chunk dict."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("text", "content", "full_text", "raw_text", "canonical_text"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val
        # Never stringify the dict — that produces garbage metadata text
        return ""
    if hasattr(item, "text"):
        val = getattr(item, "text", "")
        if isinstance(val, str) and val.strip():
            return val
    if hasattr(item, "full_text"):
        val = getattr(item, "full_text", "")
        if isinstance(val, str) and val.strip():
            return val
    # Never return str(item) — that's the garbage source
    return ""

def train_on_document(text, subscription_id, profile_id, doc_tag, doc_name, device: Optional[str] = None):
    """
     COMPLETELY FIXED: Trains and stores embeddings with strict document_id verification

    Critical fixes:
    1. Always pass doc_tag as document_id parameter
    2. Verify chunks before generating embeddings
    3. Verify metadata before saving to Qdrant
    """
    try:
        telemetry = telemetry_store() if METRICS_V2_ENABLED else None
        metrics_store = get_metrics_store()
        coverage_ratio = None
        dropped_chunks = 0
        logger.info(f"=" * 80)
        logger.info(f"Starting training for {doc_name}")
        logger.info(f"  Document ID: {doc_tag}")
        logger.info(f"  Subscription: {subscription_id}")
        logger.info(f"  Profile: {profile_id}")
        logger.info(f"=" * 80)

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
            doc_type_hint = text.get("doc_type") or text.get("document_type") or text.get("type")
            doc_domain = text.get("doc_domain")
            doc_metadata = _fetch_document_metadata(doc_tag, doc_name, doc_type_hint)
            doc_type = doc_metadata.get("doc_type") or doc_type_hint

            chunk_metadata = list(text.get("chunk_metadata") or [])
            texts = [_extract_text_from_item(t) for t in (text.get("texts") or [])]
            texts = [t for t in texts if t.strip()]

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
                        "doc_domain": doc_domain,
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
                logger.warning(f"Fixing document_id in structured data: {doc_ids} -> {doc_tag}")
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

            chunk_metadata = normalize_chunk_metadata(
                chunk_metadata,
                document_id=doc_tag,
                default_doc_type=doc_type,
                default_chunking_mode="section_aware",
            )
            for idx, meta in enumerate(chunk_metadata):
                meta["chunk_index"] = idx
                if doc_domain and not meta.get("doc_domain"):
                    meta["doc_domain"] = doc_domain
                if "sentence_complete" not in meta:
                    meta["sentence_complete"] = str(texts[idx]).strip().endswith((".", "?", "!"))

            texts, chunk_metadata, prep_stats, rescued_fragments = prepare_embedding_chunks(
                texts,
                chunk_metadata,
                subscription_id=subscription_id,
                profile_id=profile_id,
                document_id=doc_tag,
                doc_name=doc_name,
                quality_filter=_apply_chunk_quality_filter,
            )
            dropped_chunks = prep_stats.dropped_quality + prep_stats.dropped_dedupe
            if not texts:
                raise ValueError(f"Chunk preparation removed all chunks for document {doc_tag}")

            # Update full_text with clean content from prepared chunks so fallback
            # paths never re-chunk garbage ExtractedDocument repr strings.
            text["full_text"] = "\n\n".join(texts)

            min_chars = int(getattr(Config.Retrieval, "MIN_CHARS", 80))
            min_tokens = int(getattr(Config.Retrieval, "MIN_TOKENS", 15))
            min_valid = int(getattr(Config.Retrieval, "MIN_REQUIRED_CHUNKS", 3))
            texts, chunk_metadata, validity_diag, valid_indices = _filter_invalid_chunk_texts(
                texts,
                chunk_metadata,
                min_chars=min_chars,
                min_tokens=min_tokens,
            )
            dropped_chunks += int(validity_diag.get("invalid_chunks", 0))
            force_reembed = False
            if len(texts) < min_valid:
                full_text = text.get("full_text") or text.get("text") or text.get("content")
                fallback_used = False
                if is_valid_text(full_text, min_chars=min_chars, min_tokens=min_tokens):
                    fallback_chunks, fallback_meta, fallback_prep, fallback_diag, fallback_indices = (
                        _fallback_chunks_for_full_text(
                            str(full_text),
                            doc_tag=doc_tag,
                            doc_name=doc_name,
                            subscription_id=subscription_id,
                            profile_id=profile_id,
                            doc_type=doc_type,
                            doc_ocr_confidence=None,
                            min_required=min_valid,
                            min_chars=min_chars,
                            min_tokens=min_tokens,
                        )
                    )
                    if len(fallback_chunks) >= min_valid:
                        texts = fallback_chunks
                        chunk_metadata = fallback_meta
                        validity_diag = fallback_diag or validity_diag
                        valid_indices = fallback_indices
                        dropped_chunks = (
                            int((fallback_prep or ChunkPrepStats()).dropped_quality)
                            + int((fallback_prep or ChunkPrepStats()).dropped_dedupe)
                            + int(validity_diag.get("invalid_chunks", 0))
                        )
                        fallback_used = True
                        force_reembed = True

                if len(texts) < min_valid:
                    extracted_chars = len(str(full_text or "").strip())
                    diagnostics = {
                        **(validity_diag or {}),
                        "reason": "insufficient_valid_chunks",
                        "extracted_chars": extracted_chars,
                        "expected_chunks": int(validity_diag.get("total_chunks", 0)),
                        "prepared_chunks": int(validity_diag.get("total_chunks", 0)),
                        "valid_chunks": len(texts),
                        "min_required": min_valid,
                        "chunking_mode": "sliding_window_fallback" if fallback_used else "section_aware",
                        "fallback_used": fallback_used,
                    }
                    _mark_chunking_failed(doc_tag, doc_name, diagnostics)
                    raise ChunkingDiagnosticError(
                        f"EXTRACTION_OR_CHUNKING_FAILED: valid_chunks={len(texts)} "
                        f"min_required={min_valid} for document {doc_tag}",
                        diagnostics=diagnostics,
                    )
            section_intel_result = None
            if texts:
                try:
                    from src.intelligence.section_intelligence_builder import SectionIntelligenceBuilder

                    full_text = text.get("full_text") or text.get("text") or text.get("content")
                    full_text = full_text if isinstance(full_text, str) else " ".join(texts)
                    builder = SectionIntelligenceBuilder()
                    section_intel_result = builder.build(
                        document_id=str(doc_tag),
                        document_text=full_text or " ".join(texts),
                        chunk_texts=list(texts),
                        chunk_metadata=chunk_metadata,
                        metadata={
                            "doc_type": doc_type or document_type,
                            "source_name": doc_metadata.get("filename") or doc_name,
                        },
                    )
                    text["section_intelligence"] = section_intel_result.to_dict()
                    if section_intel_result.doc_domain:
                        text["doc_domain"] = section_intel_result.doc_domain
                    if getattr(Config.Intelligence, "SECTION_SUMMARY_VECTORS_ENABLED", False):
                        max_chars = int(getattr(Config.Intelligence, "SECTION_SUMMARY_MAX_CHARS", 700))
                        added = 0
                        for section in section_intel_result.sections:
                            summary_text = section_intel_result.section_summaries.get(section.section_id)
                            if not summary_text:
                                continue
                            summary_text = summary_text[:max_chars].strip()
                            if not summary_text or len(summary_text.split()) < 6:
                                continue
                            page_start = None
                            page_end = None
                            if section.page_range:
                                page_start, page_end = section.page_range
                            chunk_metadata.append(
                                {
                                    "document_id": doc_tag,
                                    "section_id": section.section_id,
                                    "section_title": section.section_title,
                                    "section_path": section.section_path,
                                    "section_kind": section.section_kind,
                                    "page_start": page_start,
                                    "page_end": page_end,
                                    "page_number": page_start,
                                    "chunk_type": "summary",
                                    "chunk_kind": "section_summary",
                                    "chunking_mode": "section_summary",
                                    "doc_type": doc_type,
                                    "sentence_complete": True,
                                }
                            )
                            texts.append(summary_text)
                            added += 1
                        if added:
                            force_reembed = True
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Section intelligence build skipped for %s: %s", doctag, exc)
            chunk_metadata = normalize_chunk_chain(
                chunk_metadata,
                subscription_id=subscription_id,
                profile_id=profile_id,
                document_id=doc_tag,
                chunks=texts,
            )

            chunk_lengths = [len(t) for t in texts]
            sentence_flags = [bool(meta.get("sentence_complete", False)) for meta in chunk_metadata]
            _record_chunking_metrics(
                metrics_store=metrics_store,
                doc_tag=doc_tag,
                chunk_lengths=chunk_lengths,
                sentence_complete_flags=sentence_flags,
            )

            if force_reembed or (
                isinstance(text.get("embeddings"), (list, tuple))
                and len(text.get("embeddings") or []) != len(texts)
            ):
                logger.info(
                    "Re-embedding structured chunks after integrity/dedupe: %s -> %s",
                    len(text.get("embeddings") or []),
                    len(texts),
                )
                _encode_start = time.time()
                text["embeddings"] = encode_with_fallback(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    device=device,
                )
                _encode_elapsed = time.time() - _encode_start
                _cps = len(texts) / max(_encode_elapsed, 0.001)
                logger.info(
                    "Embedding encode completed for %s: %d chunks in %.1fs (%.1f chunks/sec)",
                    doc_tag, len(texts), _encode_elapsed, _cps,
                )
                text["sparse_vectors"] = build_sparse_vectors(texts)
            else:
                if isinstance(text.get("embeddings"), (list, tuple)) and len(text.get("embeddings") or []) == len(valid_indices):
                    text["embeddings"] = [text["embeddings"][i] for i in valid_indices]
                if isinstance(text.get("sparse_vectors"), (list, tuple)) and len(text.get("sparse_vectors") or []) == len(valid_indices):
                    text["sparse_vectors"] = [text["sparse_vectors"][i] for i in valid_indices]
                if isinstance(text.get("summaries"), (list, tuple)) and len(text.get("summaries") or []) == len(valid_indices):
                    text["summaries"] = [text["summaries"][i] for i in valid_indices]
                if isinstance(text.get("pages"), (list, tuple)) and len(text.get("pages") or []) == len(valid_indices):
                    text["pages"] = [text["pages"][i] for i in valid_indices]
                if isinstance(text.get("sections"), (list, tuple)) and len(text.get("sections") or []) == len(valid_indices):
                    text["sections"] = [text["sections"][i] for i in valid_indices]

            text["texts"] = texts
            text["chunk_metadata"] = chunk_metadata
            text["doc_type"] = doc_type
            # Merge pickle-injected understanding into MongoDB metadata
            _pickle_meta = text.get("doc_metadata") or {}
            if isinstance(_pickle_meta, dict):
                for _ukey in ("document_summary", "key_entities", "key_facts", "intent_tags"):
                    if _pickle_meta.get(_ukey) and not doc_metadata.get(_ukey):
                        doc_metadata[_ukey] = _pickle_meta[_ukey]
            text["doc_metadata"] = doc_metadata

            pre_upsert_count = len(texts)
            result = save_embeddings_to_qdrant(text, subscription_id, profile_id, doc_tag, doc_name)
            saved = result.get("points_saved", 0)
            upsert_dropped = int(result.get("dropped_invalid", 0))
            upsert_dedup = int(result.get("dropped_dedup", 0))
            dropped_chunks += upsert_dropped + upsert_dedup
            # Post-pipeline expected count accounts for chunks dropped during upsert
            expected_points = pre_upsert_count - upsert_dropped - upsert_dedup
            if upsert_dropped:
                logger.warning(
                    "Embedding upsert dropped %d/%d chunks for %s (below min_chars/min_tokens)",
                    upsert_dropped, pre_upsert_count, doc_tag,
                )
            if expected_points and saved != expected_points:
                raise ValueError(
                    f"Embedding upsert mismatch for {doc_tag}: expected {expected_points}, saved {saved}"
                )
            logger.info(f" Stored {saved} structured embeddings")
            return {
                "status": "success",
                "points_saved": saved,
                "chunks": pre_upsert_count,
                "dropped_chunks": upsert_dropped + upsert_dedup,  # only upsert-level drops; prepare/validity already excluded from pre_upsert_count
                "coverage_ratio": None,
            }
        elif isinstance(text, ExtractedDocument):
            doc_type = text.doc_type
            ocr_confidences = (text.metrics or {}).get("ocr_confidences", []) if text.metrics else []
            doc_ocr_confidence = None
            if ocr_confidences:
                try:
                    doc_ocr_confidence = float(sum(ocr_confidences) / len(ocr_confidences))
                except Exception as exc:
                    logger.debug("OCR confidence calculation failed: %s", exc)
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

            layout_used = False
            section_intel_result = None
            doc_domain = None
            layout_graph_cache = None
            chunking_mode = "layout_graph"
            coverage_ratio = None
            try:
                from src.embedding.layout_semantics import build_semantic_payloads
                from src.api.layout_graph_store import load_layout_graph

                layout_graph = load_layout_graph(doc_tag)
                layout_graph_cache = layout_graph
                semantic = build_semantic_payloads(
                    layout_graph=layout_graph,
                    extracted=text,
                    document_id=str(doc_tag),
                    source_name=doc_name,
                )
                chunks = semantic.chunks
                chunk_metadata = semantic.chunk_metadata
                doc_domain = semantic.doc_domain
                section_intel_result = {
                    "doc_domain": doc_domain,
                    "sections": semantic.sections,
                    "section_facts": semantic.entity_facts,
                    "section_summaries": semantic.section_summaries,
                    "doc_summary": semantic.doc_summary,
                }
                layout_used = True
                full_text = normalize_text(text.full_text or "")
                if full_text:
                    coverage_ratio = len("".join([normalize_text(c) for c in chunks if c])) / max(1, len(full_text))
            except Exception as exc:  # noqa: BLE001
                logger.warning("LayoutGraph chunking failed for %s; falling back: %s", doc_name, exc)
                chunking_mode = "section_aware"
                chunks, chunk_metadata, coverage_ratio = _chunk_with_section_chunker(
                    text,
                    doc_tag=doc_tag,
                    doc_name=doc_name,
                    doc_type=doc_type,
                    doc_ocr_confidence=doc_ocr_confidence,
                    chunking_mode=chunking_mode,
                )

            if layout_used and section_intel_result:
                try:
                    from src.api.dw_newron import get_redis_client
                    from src.intelligence.redis_intel_cache import RedisIntelCache

                    redis_client = get_redis_client()
                    cache = RedisIntelCache(redis_client)
                    cache.set_json(
                        cache.layout_key(str(subscription_id), str(profile_id), str(doc_tag)),
                        {
                            "document_id": str(doc_tag),
                            "subscription_id": str(subscription_id),
                            "profile_id": str(profile_id),
                            "layout_graph": layout_graph_cache,
                            "section_summaries": section_intel_result.get("section_summaries") or {},
                            "doc_summary": section_intel_result.get("doc_summary") or "",
                            "updated_at": time.time(),
                        },
                        ttl_seconds=cache.summary_ttl,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("LayoutGraph cache skipped for %s: %s", doc_tag, exc)

            full_text = normalize_text(text.full_text or "")
            coverage_threshold = float(getattr(Config.Retrieval, "CHUNK_COVERAGE_THRESHOLD", 0.98))
            if full_text and coverage_ratio is not None and coverage_ratio < coverage_threshold:
                logger.warning(
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
                    chunking_mode=chunking_mode,
                )
                coverage_ratio = len("".join(chunks)) / max(1, len(full_text)) if full_text else coverage_ratio

            if not chunks:
                raise ValueError(f"No valid chunks extracted for {doc_name}")

            for meta in chunk_metadata:
                meta.setdefault("chunk_kind", meta.get("chunk_type") or "section_text")

            chunks, chunk_metadata, prep_stats, rescued_fragments = prepare_embedding_chunks(
                chunks,
                chunk_metadata,
                subscription_id=subscription_id,
                profile_id=profile_id,
                document_id=doc_tag,
                doc_name=doc_name,
                quality_filter=_apply_chunk_quality_filter,
            )
            dropped_chunks = prep_stats.dropped_quality + prep_stats.dropped_dedupe
            if not chunks:
                raise ValueError(f"Chunk preparation removed all chunks for {doc_name}")

            min_chars = int(getattr(Config.Retrieval, "MIN_CHARS", 80))
            min_tokens = int(getattr(Config.Retrieval, "MIN_TOKENS", 15))
            min_valid = int(getattr(Config.Retrieval, "MIN_REQUIRED_CHUNKS", 3))
            chunks, chunk_metadata, validity_diag, _valid_indices = _filter_invalid_chunk_texts(
                chunks,
                chunk_metadata,
                min_chars=min_chars,
                min_tokens=min_tokens,
            )
            dropped_chunks += int(validity_diag.get("invalid_chunks", 0))
            fallback_used = False
            if len(chunks) < min_valid:
                full_text = normalize_text(text.full_text or "")
                if is_valid_text(full_text, min_chars=min_chars, min_tokens=min_tokens):
                    fallback_chunks, fallback_meta, fallback_prep, fallback_diag, _fallback_indices = (
                        _fallback_chunks_for_full_text(
                            full_text,
                            doc_tag=doc_tag,
                            doc_name=doc_name,
                            subscription_id=subscription_id,
                            profile_id=profile_id,
                            doc_type=doc_type,
                            doc_ocr_confidence=doc_ocr_confidence,
                            min_required=min_valid,
                            min_chars=min_chars,
                            min_tokens=min_tokens,
                        )
                    )
                    if len(fallback_chunks) >= min_valid:
                        chunks = fallback_chunks
                        chunk_metadata = fallback_meta
                        validity_diag = fallback_diag or validity_diag
                        dropped_chunks = (
                            int((fallback_prep or ChunkPrepStats()).dropped_quality)
                            + int((fallback_prep or ChunkPrepStats()).dropped_dedupe)
                            + int(validity_diag.get("invalid_chunks", 0))
                        )
                        chunking_mode = "sliding_window_fallback"
                        fallback_used = True

                if len(chunks) < min_valid:
                    diagnostics = {
                        **(validity_diag or {}),
                        "reason": "insufficient_valid_chunks",
                        "extracted_chars": len(str(text or "").strip()),
                        "expected_chunks": int(validity_diag.get("total_chunks", 0)),
                        "prepared_chunks": int(validity_diag.get("total_chunks", 0)),
                        "valid_chunks": len(chunks),
                        "min_required": min_valid,
                        "chunking_mode": "sliding_window_fallback" if fallback_used else "section_aware",
                        "fallback_used": fallback_used,
                    }
                    _mark_chunking_failed(doc_tag, doc_name, diagnostics)
                    raise ChunkingDiagnosticError(
                        f"EXTRACTION_OR_CHUNKING_FAILED: valid_chunks={len(chunks)} "
                        f"min_required={min_valid} for document {doc_tag}",
                        diagnostics=diagnostics,
                    )

            chunk_lengths = [len(chunk) for chunk in chunks]
            sentence_flags = [bool(meta.get("sentence_complete", False)) for meta in chunk_metadata]
            _record_chunking_metrics(
                metrics_store=metrics_store,
                doc_tag=doc_tag,
                chunk_lengths=chunk_lengths,
                sentence_complete_flags=sentence_flags,
            )

            logger.info("Generated %s section-aware chunks for %s", len(chunks), doc_name)

            # Pre-embedding validation: filter garbage text before vectorization
            from src.embedding.pipeline.schema_normalizer import _is_metadata_garbage as _is_embed_garbage
            _valid_c, _valid_m = [], []
            for _vc_t, _vc_m in zip(chunks, chunk_metadata):
                if _is_embed_garbage(_vc_t) or len((_vc_t or "").strip()) < 20:
                    logger.warning("Pre-embed gate: dropping garbage chunk for %s", doc_tag)
                    continue
                _valid_c.append(_vc_t)
                _valid_m.append(_vc_m)
            if _valid_c:
                chunks = _valid_c
                chunk_metadata = _valid_m

            embed_start = time.time()
            embeddings_array = encode_with_fallback(
                chunks,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=device,
            )
            embed_latency_ms = (time.time() - embed_start) * 1000
            logger.info(
                "Embedding encode completed for %s: %d chunks in %.1fs (%.1f chunks/sec)",
                doc_tag, len(chunks), embed_latency_ms / 1000.0,
                len(chunks) / max(embed_latency_ms / 1000.0, 0.001),
            )
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
            try:
                summaries = compute_section_summaries(chunks, chunk_metadata, extracted=text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Section summaries skipped for %s: %s", doc_tag, exc)
                summaries = [None for _ in chunks]

            ctx = ContextUnderstanding()
            doc_summary_text = ""
            section_summary_map: Dict[str, str] = {}
            if section_intel_result and isinstance(section_intel_result, dict):
                doc_summary_text = (section_intel_result.get("doc_summary") or "").strip()
                section_summary_map = section_intel_result.get("section_summaries") or {}
            if not doc_summary_text:
                try:
                    doc_summary_bundle = ctx.summarize_document(text)
                    doc_summary_text = (doc_summary_bundle.get("abstract") or "").strip()
                    if not section_summary_map:
                        section_summary_map = doc_summary_bundle.get("section_summaries") or {}
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Document summary skipped for %s: %s", doc_tag, exc)
            section_title_to_pages = {
                (sec.title or "").strip(): (sec.start_page, sec.end_page) for sec in text.sections
            }

            extra_chunks: List[str] = []
            extra_meta: List[Dict[str, Any]] = []

            if doc_summary_text and not layout_used:
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
                        "chunking_mode": chunking_mode,
                        "doc_type": doc_type,
                        "sentence_complete": True,
                    }
                )

            if not layout_used:
                for title, summary in section_summary_map.items():
                    if not getattr(Config.Intelligence, "SECTION_SUMMARY_VECTORS_ENABLED", False):
                        continue
                    summary_text = str(summary).strip()
                    if not summary_text or len(summary_text.split()) < 6:
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
                            "chunking_mode": chunking_mode,
                            "doc_type": doc_type,
                            "sentence_complete": True,
                        }
                    )

            if not layout_used:
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
                            "chunking_mode": chunking_mode,
                            "doc_type": doc_type,
                            "sentence_complete": True,
                        }
                    )

            if not layout_used:
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
                                "chunking_mode": chunking_mode,
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
                                "chunking_mode": chunking_mode,
                                "doc_type": doc_type,
                                "sentence_complete": True,
                            }
                        )

            if extra_chunks:
                extra_chunks, extra_meta, dropped_extra = _apply_chunk_quality_filter(extra_chunks, extra_meta)
                if extra_chunks:
                    extra_chunks, extra_meta, extra_diag, _valid_indices = _filter_invalid_chunk_texts(
                        extra_chunks,
                        extra_meta,
                        min_chars=min_chars,
                        min_tokens=min_tokens,
                    )
                    dropped_extra += int(extra_diag.get("invalid_chunks", 0))
                for idx, meta in enumerate(extra_meta):
                    meta["chunk_index"] = len(chunks) + idx
                dropped_chunks += dropped_extra
                if extra_chunks:
                    chunks.extend(extra_chunks)
                    chunk_metadata.extend(extra_meta)
                    sparse_vectors.extend(build_sparse_vectors(extra_chunks))
                    extra_embeddings = encode_with_fallback(
                        extra_chunks,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        device=device,
                    )
                    embeddings_array = list(embeddings_array) + list(extra_embeddings)
                    summaries.extend([None for _ in extra_chunks])

            for meta in chunk_metadata:
                section_title = (meta.get("section_title") or "Untitled Section").strip() or "Untitled Section"
                section_path = (meta.get("section_path") or section_title).strip() or section_title
                meta["section_title"] = section_title
                meta["section_path"] = section_path
                meta["document_id"] = doc_tag
                meta.setdefault("doc_type", doc_type)
                if doc_domain:
                    meta.setdefault("doc_domain", doc_domain)
                meta.setdefault("chunking_mode", chunking_mode)
                meta["section_id"] = meta.get("section_id") or hashlib.sha1(
                    f"{doc_tag}|{section_path}".encode("utf-8")
                ).hexdigest()[:12]

            chunk_metadata = normalize_chunk_chain(
                chunk_metadata,
                subscription_id=subscription_id,
                profile_id=profile_id,
                document_id=doc_tag,
                chunks=chunks,
            )

            embeddings_payload = {
                "embeddings": embeddings_array,
                "texts": chunks,
                "sparse_vectors": sparse_vectors,
                "summaries": summaries,
                "chunk_metadata": chunk_metadata,
                "doc_metadata": doc_metadata,
                "doc_type": doc_type,
            }

            # Save to Qdrant (will perform additional verification)
            logger.info(f"Saving to Qdrant...")
            result = save_embeddings_to_qdrant(
                embeddings_payload,
                subscription_id,
                profile_id,
                doc_tag,
                doc_name
            )

            logger.info(f"=" * 80)
            saved = result.get("points_saved", 0)
            upsert_dropped_ed = int(result.get("dropped_invalid", 0))
            upsert_dedup_ed = int(result.get("dropped_dedup", 0))
            dropped_chunks += upsert_dropped_ed + upsert_dedup_ed
            expected_points = len(chunks) - upsert_dropped_ed - upsert_dedup_ed
            if expected_points and saved != expected_points:
                raise ValueError(
                    f"Embedding upsert mismatch for {doc_tag}: expected {expected_points}, saved {saved}"
                )
            logger.info(f" SUCCESS: Stored {saved} embeddings")
            logger.info(f"  Document ID: {doc_tag}")
            logger.info(f"  File: {doc_name}")
            logger.info(f"=" * 80)

            return {
                "status": "success",
                "points_saved": saved,
                "chunks": len(chunks),  # pre-dedup count
                "dropped_chunks": upsert_dropped_ed + upsert_dedup_ed,  # only upsert-level drops
                "coverage_ratio": coverage_ratio,
            }

        elif isinstance(text, str):
            if not text.strip():
                raise ValueError(f"Empty content in {doc_name}")

            logger.info(f"Chunking document with document_id={doc_tag}")

            doc_metadata = _fetch_document_metadata(doc_tag, doc_name, None)
            doc_type = doc_metadata.get("doc_type")

            chunking_mode = "section_aware"
            try:
                chunks, chunk_metadata, _ = _chunk_with_section_chunker(
                    text,
                    doc_tag=doc_tag,
                    doc_name=doc_name,
                    doc_type=doc_type,
                    doc_ocr_confidence=None,
                    chunking_mode=chunking_mode,
                )
            except Exception as exc:  # noqa: BLE001
                # Fallback to the legacy chunker to avoid total failure on edge cases.
                logger.warning("Section chunking failed for %s: %s; falling back", doc_name, exc)
                chunks_with_meta = chunk_text_for_embedding(text, doc_name, document_id=doc_tag)
                chunks = [chunk_text for chunk_text, _meta in chunks_with_meta if (chunk_text or "").strip()]
                chunk_metadata = [meta for chunk_text, meta in chunks_with_meta if (chunk_text or "").strip()]
                chunking_mode = "sliding_window_fallback"

            if not chunks:
                raise ValueError(f"No valid chunks in {doc_name}")

            chunks, chunk_metadata, prep_stats, rescued_fragments = prepare_embedding_chunks(
                chunks,
                chunk_metadata,
                subscription_id=subscription_id,
                profile_id=profile_id,
                document_id=doc_tag,
                doc_name=doc_name,
                quality_filter=_apply_chunk_quality_filter,
            )
            dropped_chunks = prep_stats.dropped_quality + prep_stats.dropped_dedupe
            if not chunks:
                raise ValueError(f"Chunk preparation removed all chunks for {doc_name}")

            min_chars = int(getattr(Config.Retrieval, "MIN_CHARS", 80))
            min_tokens = int(getattr(Config.Retrieval, "MIN_TOKENS", 15))
            min_valid = int(getattr(Config.Retrieval, "MIN_REQUIRED_CHUNKS", 3))
            chunks, chunk_metadata, validity_diag, _valid_indices = _filter_invalid_chunk_texts(
                chunks,
                chunk_metadata,
                min_chars=min_chars,
                min_tokens=min_tokens,
            )
            dropped_chunks += int(validity_diag.get("invalid_chunks", 0))
            fallback_used = False
            if len(chunks) < min_valid:
                if is_valid_text(text, min_chars=min_chars, min_tokens=min_tokens):
                    fallback_chunks, fallback_meta, fallback_prep, fallback_diag, _fallback_indices = (
                        _fallback_chunks_for_full_text(
                            text,
                            doc_tag=doc_tag,
                            doc_name=doc_name,
                            subscription_id=subscription_id,
                            profile_id=profile_id,
                            doc_type=doc_type,
                            doc_ocr_confidence=None,
                            min_required=min_valid,
                            min_chars=min_chars,
                            min_tokens=min_tokens,
                        )
                    )
                    if len(fallback_chunks) >= min_valid:
                        chunks = fallback_chunks
                        chunk_metadata = fallback_meta
                        validity_diag = fallback_diag or validity_diag
                        dropped_chunks = (
                            int((fallback_prep or ChunkPrepStats()).dropped_quality)
                            + int((fallback_prep or ChunkPrepStats()).dropped_dedupe)
                            + int(validity_diag.get("invalid_chunks", 0))
                        )
                        chunking_mode = "sliding_window_fallback"
                        fallback_used = True

                if len(chunks) < min_valid:
                    diagnostics = {
                        **(validity_diag or {}),
                        "reason": "insufficient_valid_chunks",
                        "extracted_chars": len(str(text or "").strip()),
                        "expected_chunks": int(validity_diag.get("total_chunks", 0)),
                        "prepared_chunks": int(validity_diag.get("total_chunks", 0)),
                        "valid_chunks": len(chunks),
                        "min_required": min_valid,
                        "chunking_mode": "sliding_window_fallback" if fallback_used else "section_aware",
                        "fallback_used": fallback_used,
                    }
                    _mark_chunking_failed(doc_tag, doc_name, diagnostics)
                    raise ChunkingDiagnosticError(
                        f"EXTRACTION_OR_CHUNKING_FAILED: valid_chunks={len(chunks)} "
                        f"min_required={min_valid} for document {doc_tag}",
                        diagnostics=diagnostics,
                    )

            chunk_lengths = [len(chunk) for chunk in chunks]
            sentence_flags = [bool(meta.get("sentence_complete", False)) for meta in chunk_metadata]
            _record_chunking_metrics(
                metrics_store=metrics_store,
                doc_tag=doc_tag,
                chunk_lengths=chunk_lengths,
                sentence_complete_flags=sentence_flags,
            )

            logger.info("Generated %s section-aware chunks for %s", len(chunks), doc_name)

            # Pre-embedding validation: filter garbage text before vectorization
            from src.embedding.pipeline.schema_normalizer import _is_metadata_garbage as _is_embed_garbage
            _valid_c, _valid_m = [], []
            for _vc_t, _vc_m in zip(chunks, chunk_metadata):
                if _is_embed_garbage(_vc_t) or len((_vc_t or "").strip()) < 20:
                    logger.warning("Pre-embed gate: dropping garbage chunk for %s", doc_tag)
                    continue
                _valid_c.append(_vc_t)
                _valid_m.append(_vc_m)
            if _valid_c:
                chunks = _valid_c
                chunk_metadata = _valid_m

            embed_start = time.time()
            embeddings_array = encode_with_fallback(
                chunks,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=device,
            )
            embed_latency_ms = (time.time() - embed_start) * 1000
            logger.info(
                "Embedding encode completed for %s: %d chunks in %.1fs (%.1f chunks/sec)",
                doc_tag, len(chunks), embed_latency_ms / 1000.0,
                len(chunks) / max(embed_latency_ms / 1000.0, 0.001),
            )
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
            try:
                summaries = compute_section_summaries(chunks, chunk_metadata)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Section summaries skipped for %s: %s", doc_tag, exc)
                summaries = []
            if not any(summaries):
                summaries = [chunk[:200] + "..." if len(chunk) > 200 else chunk for chunk in chunks]

            #  VERIFICATION STEP 2: Final check before saving
            logger.info(f"Final verification before saving to Qdrant")
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
                meta.setdefault("chunking_mode", chunking_mode)
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
                # chunk_id/prev/next already normalized by normalize_chunk_chain

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
            logger.info(f"Saving to Qdrant...")
            result = save_embeddings_to_qdrant(
                embeddings,
                subscription_id,
                profile_id,
                doc_tag,
                doc_name
            )

            logger.info(f"=" * 80)
            saved = result.get("points_saved", 0)
            upsert_dropped_ed = int(result.get("dropped_invalid", 0))
            upsert_dedup_ed = int(result.get("dropped_dedup", 0))
            dropped_chunks += upsert_dropped_ed + upsert_dedup_ed
            expected_points = len(chunks) - upsert_dropped_ed - upsert_dedup_ed
            if expected_points and saved != expected_points:
                raise ValueError(
                    f"Embedding upsert mismatch for {doc_tag}: expected {expected_points}, saved {saved}"
                )
            logger.info(f" SUCCESS: Stored {saved} embeddings")
            logger.info(f"  Document ID: {doc_tag}")
            logger.info(f"  File: {doc_name}")
            logger.info(f"=" * 80)

            return {
                "status": "success",
                "points_saved": saved,
                "chunks": len(chunks),  # pre-dedup count
                "dropped_chunks": upsert_dropped_ed + upsert_dedup_ed,  # only upsert-level drops
                "coverage_ratio": coverage_ratio,
            }

        else:
            raise ValueError(f"Unsupported format: {type(text)}")

    except Exception as e:
        logger.error(f"=" * 80)
        logger.error(f"L TRAINING FAILED for {doc_name}")
        logger.error(f"  Document ID: {doc_tag}")
        logger.error(f"  Error: {e}")
        logger.error(f"=" * 80)
        try:
            if telemetry:
                telemetry.increment("embedding_failures_count")
        except Exception as tel_exc:
            logger.debug("Telemetry increment failed: %s", tel_exc)
        raise

def process_document_pipeline(
    document_id: str,
    file_bytes: bytes,
    filename: str,
    subscription_id: Optional[str] = None,
    profile_id: Optional[str] = None,
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
        logger.error("Document extraction failed for %s: %s", document_id, exc)
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
        logger.error("Metadata update failed for %s: %s", document_id, exc)
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
        logger.error("Security screening failed for %s: %s", document_id, exc)
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
        logger.error("PII masking failed for %s: %s", document_id, exc)
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
        logger.error("Embedding failed for %s: %s", document_id, exc)
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

    # Preserve extracted pickle files for audit / retraining. Do NOT delete by default.
    try:
        logger.info("Preserving extracted pickle for document %s (deletion skipped)", document_id)
        cleanup_info["pickle_deleted"] = False
        cleanup_info["cleanup_pending"] = False
    except Exception as exc:  # noqa: BLE001
        logger.warning("Preserve-pickle logging failed for %s: %s", document_id, exc)
        cleanup_info["pickle_deleted"] = False
        cleanup_info["cleanup_pending"] = False

    return {
        "document_id": document_id,
        "extraction": extraction_info,
        "security": security_info,
        "embedding": embedding_info,
        "cleanup": cleanup_info,
        "errors": errors,
    }

def trainData(subscription_id: str = None):
    """Extraction-only pipeline for documents eligible for processing."""
    try:
        from src.api.extraction_service import extract_documents

        logger.info("=" * 80)
        logger.info("Starting extraction process (subscription=%s)", subscription_id)
        logger.info("=" * 80)
        return extract_documents(subscription_id=subscription_id)
    except Exception as e:
        logger.error(f"Critical error in extraction data: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "documents": []}

# New function: train_single_document
def train_single_document(doc_id: str):
    """Extract a single document identified by its string ID."""
    try:
        from src.api.extraction_service import extract_single_document

        logger.info(f"Starting single-document extraction for ID: {doc_id}")
        return extract_single_document(doc_id)
    except Exception as e:
        logger.error(f"Critical error during single-document extraction for {doc_id}: {e}", exc_info=True)
        update_training_status(doc_id, 'TRAINING_FAILED', str(e))
        return {"status": "error", "message": str(e)}
