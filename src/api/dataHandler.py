
import copy
import hashlib
import json
import logging
import subprocess
import time
import uuid
from io import BytesIO
from typing import Any, Dict, List, Optional

import boto3 as b3
import numpy as np
import pandas as pd
from Crypto.Cipher import AES
from azure.storage.blob import BlobServiceClient
from bson.objectid import ObjectId
from pymongo import MongoClient, errors
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from urllib.parse import urlparse

from src.api.config import Config
from src.api.context_understanding import ContextUnderstanding
from src.api.documentVetting import mask_document_content, vettingProcessor
from src.api.dw_document_extractor import DocumentExtractor
from src.api.pipeline_models import ChunkCandidate, ChunkRecord, ExtractedDocument, Section
from src.api.vector_store import QdrantVectorStore, build_collection_name, compute_chunk_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Lazy-loaded globals to avoid heavy initialization during import
docEx = None
_MODEL = None
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
            logging.warning(f"Subscription {subscription_id} not found in database, defaulting to PII ENABLED")
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


def create_mongo_client():
    """Create a Mongo client with a graceful fallback when the primary URI is misconfigured."""
    primary_uri = Config.MongoDB.URI
    fallback_uri = getattr(Config.MongoDB, "FALLBACK_URI", None)
    try:
        client = MongoClient(primary_uri, serverSelectionTimeoutMS=5000)
        try:
            # Verify connectivity early so failures are visible
            client.admin.command('ping')
            logging.info(f"Connected to MongoDB primary URI: {primary_uri}")
        except Exception as ping_exc:
            logging.warning(f"Ping to primary MongoDB URI failed: {ping_exc}")
            # allow fallback to kick in below
            raise ping_exc
        return client
    except Exception as exc:
        if fallback_uri and fallback_uri != primary_uri:
            logging.warning(f"Primary MongoDB URI failed ({exc}); attempting fallback {fallback_uri}")
            try:
                client = MongoClient(fallback_uri, serverSelectionTimeoutMS=5000)
                client.admin.command('ping')
                logging.info(f"Connected to MongoDB fallback URI: {fallback_uri}")
                return client
            except Exception as fb_exc:
                logging.error(f"Fallback MongoDB URI also failed: {fb_exc}")
                raise
        logging.error(f"Unable to create MongoClient: {exc}")
        raise


mongoClient = create_mongo_client()
db = mongoClient[Config.MongoDB.DB]


def get_doc_extractor():
    """Lazy init for document extractor."""
    global docEx
    if docEx is None:
        docEx = DocumentExtractor()
    return docEx


def get_model():
    """Lazy init for sentence transformer model to cut import-time cost."""
    global _MODEL
    if _MODEL is None:
        name = getattr(Config.Model, "EMBEDDING_MODEL", None) or getattr(
            Config.Model, "SENTENCE_TRANSFORMERS", "BAAI/bge-large-en-v1.5"
        )
        logging.info(f"Loading sentence transformer model: {name}")
        _MODEL = SentenceTransformer(name)
        expected_dim = getattr(Config.Model, "EMBEDDING_DIM", None)
        model_dim = _MODEL.get_sentence_embedding_dimension()
        if expected_dim and model_dim != expected_dim:
            logging.warning(f"Configured EMBEDDING_DIM={expected_dim} but model dimension is {model_dim}")
        logging.info(f"Loaded sentence transformer model: {name} (dim={model_dim})")
    return _MODEL


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
                extracted_data[file_name] = extractor.extract_dataframe(df, get_model())
            elif file_name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(BytesIO(content))
                extracted_data[file_name] = extractor.extract_dataframe(df, get_model())
            elif file_name.endswith(".json"):
                extracted_data[file_name] = json.loads(content.decode("utf-8"))
            elif file_name.endswith(".pdf"):
                extracted_data[file_name] = extractor.extract_text_from_pdf(content)
            elif file_name.endswith(".docx"):
                extracted_data[file_name] = extractor.extract_text_from_docx(content)
            elif file_name.endswith((".pptx", ".ppt")):
                extracted_data[file_name] = extractor.extract_text_from_pptx(content)
            elif file_name.endswith(".txt"):
                extracted_data[file_name] = content.decode("utf-8")
            else:
                extracted_data[file_name] = content.decode("utf-8")
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


def updateVetting(document_id, new_value):
    """Updates vetting points in MongoDB."""
    try:
        filter_criteria = {"_id": ObjectId(document_id)}
        update_operation = {"$set": {'vettingPoints': new_value}}
        collection = db[Config.MongoDB.DOCUMENTS]
        result = collection.update_one(filter_criteria, update_operation)
        if result.matched_count > 0:
            logging.info(f"Document with ID {document_id} updated successfully.")
            return {"status": "success", "matched_count": result.matched_count, "modified_count": result.modified_count}
        else:
            logging.warning(f"No document found with ID {document_id}.")
            return {"status": "not_found", "matched_count": 0, "modified_count": 0}
    except Exception as e:
        logging.error(f"Error updating document: {e}")
        return {"status": "error", "message": str(e)}


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


def get_azure_docs(files):
    """Fetches documents from Azure Blob Storage."""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(Config.DocAzureBlob.AZURE_BLOB_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(Config.DocAzureBlob.AZURE_BLOB_CONTAINER_NAME)
        blob_client = container_client.get_blob_client('local/' + files)
        blob_data = blob_client.download_blob().readall()
        return blob_data
    except Exception as e:
        logging.error(f"Error fetching Azure documents: {e}")
        return None


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

        subscriptionId = str(
            docData.get('subscriptionId')
            or docData.get('subscription_id')
            or docData.get('subscription')
            or (connData.get('subscriptionId') if isinstance(connData, dict) else None)
            or (connData.get('subscription') if isinstance(connData, dict) else None)
            or "default"
        )

        # Check PII setting for this subscription
        pii_masking_enabled = get_subscription_pii_setting(subscriptionId)
        logging.info(f"Document {docId} (Subscription {subscriptionId}): PII masking = {pii_masking_enabled}")

        allowed_statuses = {'UNDER_REVIEW', 'TRAINING_FAILED'}

        if docData.get('status') in allowed_statuses:
            try:
                logging.info(f"=" * 80)
                logging.info(f"Processing document {docId}: {docData.get('name', 'Unknown')}")
                logging.info(f"=" * 80)
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
                        # Extract the path after 'az://' prefix
                        file_key = file_path.split('/', 4)[-1] if file_path.startswith('az://') else file_path
                        logging.info(f"Reading file: {file_key} for document {docId}")

                        docContent = get_azure_docs(file_key)
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

                # Apply PII masking and vetting
                if all_extracted_docs:
                    if pii_masking_enabled:
                        masked_docs, pii_count, high_conf, pii_items = mask_document_content(all_extracted_docs)
                        update_pii_stats(docId, pii_count, high_conf, pii_items)

                        if high_conf:
                            logging.error(
                                f"High confidentiality content detected in document {docId}; blocking training")
                            update_training_status(docId, 'TRAINING_BLOCKED_CONFIDENTIAL',
                                                   'High confidentiality content detected')
                            continue

                        # Recompute embeddings for structured data after masking
                        for fname, content in masked_docs.items():
                            if isinstance(content, dict) and "texts" in content:
                                texts = content.get("texts") or []
                                if texts:
                                    model = get_model()
                                    content["embeddings"] = model.encode(texts, convert_to_numpy=True)
                                masked_docs[fname] = content
                    else:
                        logging.info(f"PII masking disabled for subscription {subscriptionId}")
                        masked_docs = all_extracted_docs
                        pii_count = 0
                        high_conf = False
                        pii_items = []
                        update_pii_stats(docId, 0, False, [])

                    vettingPoints = vettingProcessor(masked_docs)
                    updateVetting(docId, vettingPoints)

                    #  Store with explicit documentId
                    dataDict[docId] = {
                        'subscriptionId': subscriptionId,
                        'profileId': profileId,
                        'documentId': docId,  #  Explicit document_id
                        'extractedDoc': masked_docs,
                        'docName': docData.get('name', 'Unknown')
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



def collectionConnect(name):
    """Fetches documents from a MongoDB collection."""
    logging.info(f"Fetching connection details for collection: {name}")
    try:
        collection = db[name]
        try:
            # Count documents to make emptiness explicit in the logs
            count = collection.count_documents({})
            logging.info(f"Collection '{name}' document count: {count}")
        except Exception as count_exc:
            logging.warning(f"Unable to count documents for collection '{name}': {count_exc}")
        return collection.find()
    except Exception as e:
        logging.error(f"Error connecting to collection {name}: {e}")
        # return an empty iterator to keep calling code behavior predictable
        return []


def extract_document_info():
    """Retrieves connector details from MongoDB."""
    try:
        logging.info(
            f"Extracting document info from DB: {Config.MongoDB.DB}, collections: {Config.MongoDB.DOCUMENTS}, {Config.MongoDB.CONNECTOR}")
        try:
            existing = db.list_collection_names()
            logging.info(f"Existing collections in DB '{Config.MongoDB.DB}': {existing}")
        except Exception as lc_exc:
            logging.warning(f"Could not list collections: {lc_exc}")

        docs = collectionConnect(Config.MongoDB.DOCUMENTS)
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
        connectors = collectionConnect(Config.MongoDB.CONNECTOR)
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
        return {}


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
            model = get_model()
            raw_vectors = model.encode(raw_vectors, convert_to_numpy=True, normalize_embeddings=True)

        normalized_vectors, vector_size = normalize_embedding_matrix(raw_vectors)
        if vector_size == 0:
            raise ValueError("Empty embeddings array")

        texts = embeddings.get("texts") or []
        chunk_metadata = embeddings.get("chunk_metadata", []) or []
        pages = embeddings.get("pages") or []
        sections = embeddings.get("sections") or []
        summaries = embeddings.get("summaries") or []
        sparse_vectors = embeddings.get("sparse_vectors") or []

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

            sparse_vector = None
            if idx < len(sparse_vectors):
                sv = sparse_vectors[idx]
                if sv.get("indices") and sv.get("values"):
                    sparse_vector = sv

            payload = {
                "subscription_id": str(subscription_id),
                "profile_id": str(profile_id),
                "text": text,
                "document_id": str(doctag),
                "source_file": source_filename,
                "chunk_index": idx,
                "chunk_count": max_len,
                "page": page_val,
                "section_title": section_val or chunk_meta.get("section"),
                "summary": summary_val,
                "chunk_id": chunk_id,
                "prev_chunk_id": chunk_meta.get("prev_chunk_id"),
                "next_chunk_id": chunk_meta.get("next_chunk_id"),
                "chunk_type": chunk_meta.get("chunk_type", "text"),
                "section_id": chunk_meta.get("section_id"),
            }

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

from src.api.enhanced_retrieval import chunk_text_for_embedding


def train_on_document(text, subscription_id, profile_id, doc_tag, doc_name):
    """
     COMPLETELY FIXED: Trains and stores embeddings with strict document_id verification

    Critical fixes:
    1. Always pass doc_tag as document_id parameter
    2. Verify chunks before generating embeddings
    3. Verify metadata before saving to Qdrant
    """
    try:
        logging.info(f"=" * 80)
        logging.info(f"Starting training for {doc_name}")
        logging.info(f"  Document ID: {doc_tag}")
        logging.info(f"  Subscription: {subscription_id}")
        logging.info(f"  Profile: {profile_id}")
        logging.info(f"=" * 80)

        if not profile_id:
            raise ValueError("profile_id is required for training")

        if isinstance(text, dict):
            # Handle pre-embedded structured data (CSV/Excel)
            #  Verify document_id in structured data
            if 'chunk_metadata' in text:
                doc_ids = set(meta.get('document_id') for meta in text['chunk_metadata'])
                doc_ids.discard(None)
                if len(doc_ids) > 1:
                    raise ValueError(f"Multiple document_ids in structured data: {doc_ids}")
                if doc_ids and list(doc_ids)[0] != doc_tag:
                    logging.warning(f"Fixing document_id in structured data: {doc_ids} -> {doc_tag}")
                    for meta in text['chunk_metadata']:
                        meta['document_id'] = doc_tag

            result = save_embeddings_to_qdrant(
                text, subscription_id, profile_id, doc_tag, doc_name
            )
            logging.info(f" Stored {result.get('points_saved', 0)} structured embeddings")
            return f"Stored {result.get('points_saved', 0)} embeddings"
        elif isinstance(text, ExtractedDocument):
            candidates: List[ChunkCandidate] = text.chunk_candidates or []
            if not candidates and text.full_text:
                logging.info("No structured candidates found; falling back to raw text chunking")
                candidates = [
                    ChunkCandidate(
                        text=text.full_text,
                        page=None,
                        section_title="Document",
                        section_id=None,
                        chunk_type="text",
                    )
                ]

            def _merge_candidates(input_candidates: List[ChunkCandidate], min_len: int = 200):
                merged = []
                buffer_text = ""
                buffer_meta: Optional[ChunkCandidate] = None
                for cand in input_candidates:
                    if not cand.text:
                        continue
                    if buffer_meta is None:
                        buffer_text = cand.text.strip()
                        buffer_meta = cand
                        continue

                    if len(buffer_text) < min_len and cand.section_id == buffer_meta.section_id:
                        buffer_text = f"{buffer_text}\n{cand.text.strip()}"
                    else:
                        merged.append((buffer_text, buffer_meta))
                        buffer_text = cand.text.strip()
                        buffer_meta = cand

                if buffer_meta and buffer_text:
                    merged.append((buffer_text, buffer_meta))
                return merged

            merged_candidates = _merge_candidates(candidates)
            if not merged_candidates:
                raise ValueError(f"No chunk candidates extracted for {doc_name}")

            chunks = []
            chunk_metadata = []
            for idx, (chunk_text, cand_meta) in enumerate(merged_candidates):
                chunks.append(chunk_text)
                chunk_metadata.append(
                    {
                        "document_id": doc_tag,
                        "section_title": cand_meta.section_title,
                        "section_id": cand_meta.section_id
                        or hashlib.sha1(f"{doc_tag}|{cand_meta.section_title}".encode("utf-8")).hexdigest()[:12],
                        "chunk_index": idx,
                        "chunk_type": cand_meta.chunk_type,
                        "page_number": cand_meta.page,
                    }
                )

            logging.info(f"Generated {len(chunks)} chunks from structured extraction for {doc_name}")

            model = get_model()
            embeddings_array = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
            sparse_vectors = build_sparse_vectors(chunks)
            summaries = compute_section_summaries(chunks, chunk_metadata, extracted=text)

            for idx, meta in enumerate(chunk_metadata):
                meta["chunk_id"] = compute_chunk_id(subscription_id, profile_id, doc_tag, doc_name, idx, chunks[idx])
                meta["chunk_type"] = meta.get("chunk_type", "text")

            embeddings_payload = {
                "embeddings": embeddings_array,
                "texts": chunks,
                "sparse_vectors": sparse_vectors,
                "summaries": summaries,
                "chunk_metadata": chunk_metadata,
                "pages": [m.get("page_number") for m in chunk_metadata],
                "sections": [m.get("section_title") for m in chunk_metadata],
            }

            result = save_embeddings_to_qdrant(
                embeddings_payload, subscription_id, profile_id, doc_tag, doc_name
            )
            logging.info(f" Stored {result.get('points_saved', 0)} structured extraction embeddings")
            return f"Stored {result.get('points_saved', 0)} embeddings"

        elif isinstance(text, str):
            if not text.strip():
                raise ValueError(f"Empty content in {doc_name}")

            #  CRITICAL: Pass doc_tag as document_id (REQUIRED parameter)
            logging.info(f"Chunking document with document_id={doc_tag}")

            try:
                chunks_with_meta = chunk_text_for_embedding(
                    text,
                    doc_name,
                    document_id=doc_tag  #  REQUIRED - no default fallback
                )
            except ValueError as ve:
                logging.error(f"L Chunking failed for {doc_name}: {ve}")
                raise ValueError(f"Chunking validation failed: {ve}")

            if not chunks_with_meta:
                raise ValueError(f"No valid chunks in {doc_name}")

            #  VERIFICATION STEP 1: Check chunks have correct document_id
            logging.info(f"Verifying {len(chunks_with_meta)} chunks for document_id={doc_tag}")

            chunks = [chunk_text for chunk_text, meta in chunks_with_meta]
            chunk_metadata = [meta for chunk_text, meta in chunks_with_meta]

            # Verify all chunks have the EXACT document_id
            doc_ids_in_chunks = set(meta.get('document_id') for meta in chunk_metadata)
            doc_ids_in_chunks.discard(None)

            if len(doc_ids_in_chunks) == 0:
                raise ValueError(f"L No document_id found in chunk metadata for {doc_name}")

            if len(doc_ids_in_chunks) > 1:
                raise ValueError(
                    f"L Multiple document_ids in chunks: {doc_ids_in_chunks}. "
                    f"Expected only: {doc_tag}"
                )

            actual_doc_id = list(doc_ids_in_chunks)[0]
            if actual_doc_id != doc_tag:
                raise ValueError(
                    f"L document_id mismatch! Expected: {doc_tag}, Found: {actual_doc_id}"
                )

            logging.info(f" VERIFIED: All {len(chunks)} chunks have document_id={doc_tag}")

            # Generate embeddings
            logging.info(f"Generating embeddings for {len(chunks)} chunks")
            model = get_model()
            embeddings_array = model.encode(
                chunks,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

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
                meta["section_id"] = meta.get("section_id") or hashlib.sha1(
                    f"{doc_tag}|{meta.get('section_title') or 'section'}".encode("utf-8")
                ).hexdigest()[:12]
                meta["chunk_type"] = meta.get("chunk_type", "text")
                meta["chunk_id"] = compute_chunk_id(subscription_id, profile_id, doc_tag, doc_name, idx, chunks[idx])

            # Prepare embeddings dict with metadata
            embeddings = {
                "embeddings": embeddings_array,
                "texts": chunks,
                "sparse_vectors": sparse_vectors,
                "summaries": summaries,
                "chunk_metadata": chunk_metadata  #  All verified to have correct document_id
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
            logging.info(f" SUCCESS: Stored {result.get('points_saved', 0)} embeddings")
            logging.info(f"  Document ID: {doc_tag}")
            logging.info(f"  File: {doc_name}")
            logging.info(f"=" * 80)

            return f"Stored {result.get('points_saved', 0)} embeddings"

        else:
            raise ValueError(f"Unsupported format: {type(text)}")

    except Exception as e:
        logging.error(f"=" * 80)
        logging.error(f"L TRAINING FAILED for {doc_name}")
        logging.error(f"  Document ID: {doc_tag}")
        logging.error(f"  Error: {e}")
        logging.error(f"=" * 80)
        raise


def trainData():
    """Main training function that processes all UNDER_REVIEW documents."""
    try:
        logging.info("=" * 80)
        logging.info("Starting training process")
        logging.info("=" * 80)

        # Extract document information
        docColl = extract_document_info()

        if not docColl:
            logging.warning("No documents found to train")
            return {"status": "no_documents", "message": "No documents found for training"}

        logging.info(f"Found {len(docColl)} documents in total")

        # Filter documents eligible for training (exclude completed or deleted)
        under_review_docs = {
            doc_id: doc_info
            for doc_id, doc_info in docColl.items()
            if doc_info['dataDict'].get('status') not in {'DELETED', 'TRAINING_COMPLETED'}
        }

        # Also compute exact count for explicitly UNDER_REVIEW status for clarity
        explicitly_under_review = {
            doc_id: doc_info
            for doc_id, doc_info in docColl.items()
            if doc_info['dataDict'].get('status') == 'UNDER_REVIEW'
        }

        logging.info(
            f"Found {len(under_review_docs)} documents eligible for processing (excluding DELETED/TRAINING_COMPLETED)")
        logging.info(f"Found {len(explicitly_under_review)} documents with status == UNDER_REVIEW")

        if not under_review_docs:
            logging.warning("No documents eligible for training (excluding DELETED/TRAINING_COMPLETED)")
            return {"status": "no_documents", "message": "No documents pending training"}

        # Process documents and extract data
        resData = connectData(under_review_docs)

        if not resData:
            logging.error("No documents were successfully processed")
            return {"status": "processing_failed", "message": "All documents failed during processing"}

        logging.info(f"Successfully processed {len(resData)} documents")

        # Train each document individually
        training_results = {
            "successful": [],
            "failed": [],
            "total": len(resData)
        }

        for doc_id, doc_data in resData.items():
            try:
                profile_id = doc_data['profileId']
                subscription_id = doc_data.get('subscriptionId', 'default')
                extracted_doc = doc_data['extractedDoc']
                doc_name = doc_data.get('docName', 'Unknown')

                if not profile_id:
                    raise ValueError(f"profile_id is required for training document {doc_id}")

                logging.info("-" * 80)
                logging.info(f"Training document: {doc_name} (ID: {doc_id})")
                logging.info(f"Profile: {profile_id}")
                logging.info(f"Number of files in document: {len(extracted_doc)}")

                file_results = []
                file_errors = []

                # Train each file within the document
                for file_name, file_content in extracted_doc.items():
                    try:
                        logging.info(f"Training file: {file_name}")
                        result = train_on_document(
                            file_content,
                            subscription_id,
                            profile_id,
                            doc_id,
                            file_name
                        )
                        logging.info(result)
                        file_results.append({
                            "file_name": file_name,
                            "result": result
                        })
                    except Exception as file_error:
                        logging.error(f"Failed to train file {file_name}: {file_error}")
                        file_errors.append({
                            "file_name": file_name,
                            "error": str(file_error)
                        })

                # Update document status based on results
                if file_results and not file_errors:
                    # All files trained successfully
                    update_training_status(doc_id, 'TRAINING_COMPLETED')
                    training_results["successful"].append({
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "files_trained": len(file_results),
                        "results": file_results
                    })
                elif file_results and file_errors:
                    # Partial success
                    error_msg = f"Partial training: {len(file_results)} succeeded, {len(file_errors)} failed"
                    update_training_status(doc_id, 'TRAINING_PARTIALLY_COMPLETED', error_msg)
                    training_results["successful"].append({
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "files_trained": len(file_results),
                        "files_failed": len(file_errors),
                        "results": file_results,
                        "errors": file_errors,
                        "status": "partial"
                    })
                else:
                    # All files failed
                    error_msg = f"All {len(file_errors)} files failed to train"
                    update_training_status(doc_id, 'TRAINING_FAILED', error_msg)
                    training_results["failed"].append({
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "errors": file_errors
                    })

            except Exception as doc_error:
                logging.error(f"Failed to train document {doc_id}: {doc_error}")
                update_training_status(doc_id, 'TRAINING_FAILED', str(doc_error))
                training_results["failed"].append({
                    "doc_id": doc_id,
                    "error": str(doc_error)
                })

        logging.info("=" * 80)
        logging.info("Training process completed")
        logging.info(f"Successful: {len(training_results['successful'])}")
        logging.info(f"Failed: {len(training_results['failed'])}")
        logging.info("=" * 80)

        return {
            "status": "completed",
            "results": training_results,
            "summary": {
                "total": training_results["total"],
                "successful": len(training_results["successful"]),
                "failed": len(training_results["failed"])
            }
        }

    except Exception as e:
        logging.error(f"Critical error in training data: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "results": None
        }


# New function: train_single_document
def train_single_document(doc_id: str):
    """Train a single document identified by its string ID.

    This function will process the specific document (perform extraction and vetting)
    and then train embeddings for its files. If the document exists but is not in
    UNDER_REVIEW status, it will still be processed for training (useful for manual triggers).
    """
    try:
        logging.info(f"Starting single-document training for ID: {doc_id}")

        # Fetch all document/connector info and locate requested doc
        docColl = extract_document_info()
        if not docColl or doc_id not in docColl:
            logging.warning(f"Document {doc_id} not found in connector information")
            return {"status": "not_found", "message": f"Document {doc_id} not found"}

        # Prepare a single-document mapping for connectData. Force status to UNDER_REVIEW
        # so that the existing processing path runs.
        single_doc_info = copy.deepcopy(docColl[doc_id])
        # Ensure dataDict exists and set status to UNDER_REVIEW to force processing
        single_doc_info.setdefault('dataDict', {})
        single_doc_info['dataDict']['status'] = 'UNDER_REVIEW'

        # connectData expects a mapping of docId -> {dataDict, connDict}
        resData = connectData({doc_id: single_doc_info})

        if not resData:
            logging.error(f"Processing failed or no content extracted for document {doc_id}")
            update_training_status(doc_id, 'TRAINING_FAILED', 'Processing or extraction failed')
            return {"status": "processing_failed", "message": f"Failed to process document {doc_id}"}

        # Use same training flow as trainData but only for this document
        training_results = {"successful": [], "failed": [], "total": len(resData)}

        for d_id, doc_data in resData.items():
            try:
                profile_id = doc_data['profileId']
                subscription_id = doc_data.get('subscriptionId', 'default')
                extracted_doc = doc_data['extractedDoc']
                doc_name = doc_data.get('docName', 'Unknown')

                if not profile_id:
                    raise ValueError(f"profile_id is required for training document {d_id}")

                logging.info(f"Training document: {doc_name} (ID: {d_id})")

                file_results = []
                file_errors = []

                for file_name, file_content in extracted_doc.items():
                    try:
                        logging.info(f"Training file: {file_name}")
                        result = train_on_document(
                            file_content,
                            subscription_id,
                            profile_id,
                            d_id,
                            file_name
                        )
                        logging.info(result)
                        file_results.append({"file_name": file_name, "result": result})
                    except Exception as file_error:
                        logging.error(f"Failed to train file {file_name}: {file_error}")
                        file_errors.append({"file_name": file_name, "error": str(file_error)})

                if file_results and not file_errors:
                    update_training_status(d_id, 'TRAINING_COMPLETED')
                    training_results['successful'].append({
                        'doc_id': d_id,
                        'doc_name': doc_name,
                        'files_trained': len(file_results),
                        'results': file_results
                    })
                elif file_results and file_errors:
                    error_msg = f"Partial training: {len(file_results)} succeeded, {len(file_errors)} failed"
                    update_training_status(d_id, 'TRAINING_PARTIALLY_COMPLETED', error_msg)
                    training_results['successful'].append({
                        'doc_id': d_id,
                        'doc_name': doc_name,
                        'files_trained': len(file_results),
                        'files_failed': len(file_errors),
                        'results': file_results,
                        'errors': file_errors,
                        'status': 'partial'
                    })
                else:
                    error_msg = f"All {len(file_errors)} files failed to train"
                    update_training_status(d_id, 'TRAINING_FAILED', error_msg)
                    training_results['failed'].append({'doc_id': d_id, 'doc_name': doc_name, 'errors': file_errors})

            except Exception as doc_error:
                logging.error(f"Failed to train document {d_id}: {doc_error}")
                update_training_status(d_id, 'TRAINING_FAILED', str(doc_error))
                training_results['failed'].append({'doc_id': d_id, 'error': str(doc_error)})

        logging.info(f"Single-document training completed for {doc_id}")
        return {"status": "completed", "results": training_results}

    except Exception as e:
        logging.error(f"Critical error during single-document training for {doc_id}: {e}", exc_info=True)
        update_training_status(doc_id, 'TRAINING_FAILED', str(e))
        return {"status": "error", "message": str(e)}
