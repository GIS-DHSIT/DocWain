
import json
import uuid
import logging
import hashlib
import subprocess
import boto3 as b3
import numpy as np
import pandas as pd
from io import BytesIO
from api.config import Config
from Crypto.Cipher import AES
from pymongo import MongoClient
from urllib.parse import urlparse
from bson.objectid import ObjectId
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, PayloadSchemaType
from sentence_transformers import SentenceTransformer
from api.documentVetting import vettingProcessor
from api.dw_document_extractor import DocumentExtractor
from azure.storage.blob import BlobServiceClient
import time
import google.generativeai as genai

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize global objects
docEx = DocumentExtractor()
MODEL = SentenceTransformer(Config.Model.SENTENCE_TRANSFORMERS)
mongoClient = MongoClient(Config.MongoDB.URI)
db = mongoClient[Config.MongoDB.DB]
qdrant_client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=60)


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
        if content:
            logging.info(f"Extracting File {file_name}")
            if file_name.endswith(".csv"):
                df = pd.read_csv(BytesIO(content))
                extracted_data[file_name] = docEx.extract_dataframe(df, MODEL)
            elif file_name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(BytesIO(content))
                extracted_data[file_name] = docEx.extract_dataframe(df, MODEL)
            elif file_name.endswith(".json"):
                extracted_data[file_name] = json.loads(content.decode("utf-8"))
            elif file_name.endswith(".pdf"):
                extracted_data[file_name] = docEx.extract_text_from_pdf(content)
            elif file_name.endswith(".docx"):
                extracted_data[file_name] = docEx.extract_text_from_docx(content)
            elif file_name.endswith((".pptx", ".ppt")):
                extracted_data[file_name] = docEx.extract_text_from_pptx(content)
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


def connectData(documentConnection):
    """Processes documents and extracts data, updates vetting points."""
    dataDict = {}

    for k, v in documentConnection.items():
        docData = v['dataDict']
        connData = v['connDict']
        profileId = str(docData['profile'])
        docId = str(docData['_id'])

        if docData['status'] == 'UNDER_REVIEW':
            try:
                logging.info(f"Processing document {docId}: {docData.get('name', 'Unknown')}")

                # Initialize extracted documents dictionary for this document
                all_extracted_docs = {}

                if docData['type'] == 'S3':
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
                    vettingPoints = vettingProcessor(extractedDoc)
                    updateVetting(docId, vettingPoints)

                elif docData['type'] == 'LOCAL':
                    files = connData['locations']

                    # Process ALL files for this document
                    for file_path in files:
                        try:
                            file_key = file_path.split('/', 4)[-1]
                            logging.info(f"Processing file: {file_key} for document {docId}")

                            docContent = get_azure_docs(file_key)

                            if docContent is None:
                                logging.error(f"Failed to read Azure file {file_key} for document {docId}")
                                continue

                            extractedDoc = fileProcessor(docContent, file_path)

                            if not extractedDoc:
                                logging.error(f"Failed to extract content from file {file_key}")
                                continue

                            # Add this file's extracted content to the collection
                            all_extracted_docs.update(extractedDoc)
                            logging.info(f"Successfully extracted content from {file_key}")

                        except Exception as file_error:
                            logging.error(f"Error processing file {file_path}: {file_error}")
                            continue

                    # Calculate vetting points for all extracted documents
                    if all_extracted_docs:
                        vettingPoints = vettingProcessor(all_extracted_docs)
                        updateVetting(docId, vettingPoints)
                    else:
                        logging.error(f"No content extracted for document {docId}")
                        update_training_status(docId, 'TRAINING_FAILED', 'No content extracted from any file')
                        continue

                # Store all extracted documents for this document ID
                if all_extracted_docs:
                    dataDict[docId] = {
                        'profileId': profileId,
                        'extractedDoc': all_extracted_docs,
                        'docName': docData.get('name', 'Unknown')
                    }
                    logging.info(f"Stored {len(all_extracted_docs)} files for document {docId}")
                else:
                    logging.error(f"No documents extracted for {docId}")
                    update_training_status(docId, 'TRAINING_FAILED', 'No content extracted')

            except Exception as e:
                logging.error(f"Error processing document {docId} ({docData.get('name', 'Unknown')}): {e}")
                update_training_status(docId, 'TRAINING_FAILED', str(e))

        elif docData['status'] == 'DELETED':
            profileData = str(docData['profile'])
            delete_embeddings(profileData, docId)

    return dataDict


def collectionConnect(name):
    """Fetches documents from a MongoDB collection."""
    logging.info(f"Fetching connection details for collection: {name}")
    collectionConn = db[name]
    return collectionConn.find()


def extract_document_info():
    """Retrieves connector details from MongoDB."""
    try:
        docs = collectionConnect(Config.MongoDB.DOCUMENTS)
        Docs = {}
        for doc in docs:
            refConnector = doc['_id'].__str__()
            Docs[refConnector] = doc
        connInfo = {}
        connectors = collectionConnect(Config.MongoDB.CONNECTOR)
        for conn in connectors:
            connId = conn['_id'].__str__()
            connInfo[connId] = conn
        connList = list(connInfo.keys())
        docInfo = {}
        for docId, docData in Docs.items():
            connRef = docData['connector'].__str__()
            if connRef in connList:
                docInfo[docId] = {'dataDict': docData, 'connDict': connInfo[connRef]}
        return docInfo

    except Exception as e:
        logging.error(f"Error fetching connection details: {e}")
        return {}


def delete_embeddings(tag, file_id):
    """Delete embeddings from Qdrant for a specific file."""
    try:
        # Define the filter to find the embeddings related to the file
        filter_criteria = {
            "must": [
                {"key": "document_id", "match": {"value": file_id}}
            ]
        }

        # Perform the deletion
        qdrant_client.delete(
            collection_name=tag,
            points_selector=filter_criteria
        )
        logging.info(f"Embeddings successfully deleted for file {file_id} in collection {tag}.")
        return {"status": "success", "message": f"Embeddings deleted for file {file_id}."}
    except Exception as e:
        logging.error(f"Error deleting embeddings: {e}")
        return {"status": "error", "message": "Error deleting embeddings."}


def ensure_qdrant_collection(collection_name, vector_size=768):
    """Ensures the Qdrant collection exists with the correct settings."""
    try:
        collections = qdrant_client.get_collections().collections
        existing_collections = {col.name for col in collections}

        if collection_name not in existing_collections:
            logging.info(f"Creating Qdrant collection: {collection_name}")
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logging.info(f"Collection '{collection_name}' created successfully.")
        else:
            logging.info(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        logging.error(f"Error ensuring collection in Qdrant: {e}")
        raise


def save_embeddings_to_qdrant(embeddings, tags, doctag, source_filename, batch_size=100):
    """Ensures the collection exists and saves embeddings to Qdrant."""
    try:
        if "embeddings" not in embeddings or embeddings["embeddings"] is None:
            logging.error(f"Error: 'embeddings' key missing or None for document {doctag}!")
            raise ValueError("Embeddings data is invalid")

        embedding_array = np.asarray(embeddings["embeddings"], dtype=np.float32)
        if embedding_array.size == 0 or embedding_array.shape[1] == 0:
            logging.error(f"Error: Trying to save empty embeddings for document {doctag}!")
            raise ValueError("Empty embeddings array")

        vector_size = embedding_array.shape[1]
        ensure_qdrant_collection(tags, vector_size)

        logging.info(f"Saving embeddings to Qdrant for profile: {tags}, document: {doctag}, file: {source_filename}")

        all_points = []
        for vector, text in zip(embedding_array, embeddings["texts"]):
            if np.all(vector == 0) or np.size(vector) == 0:
                logging.warning(f"Skipping empty or invalid vector for text: {text[:50]}...")
                continue

            vector = vector.tolist()

            all_points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "tag": str(tags),
                        "text": text,
                        "document_id": str(doctag),
                        "source_file": source_filename
                    }
                )
            )

        if not all_points:
            logging.error(f"No valid points to upload for document {doctag}, file {source_filename}")
            raise ValueError("No valid embedding points generated")

        # Upload in batches
        for i in range(0, len(all_points), batch_size):
            batch = all_points[i:i + batch_size]
            qdrant_client.upsert(collection_name=tags, points=batch)
            logging.info(
                f"Uploaded batch {i // batch_size + 1}/{(len(all_points) + batch_size - 1) // batch_size} for {source_filename}")

        logging.info(f"Successfully saved {len(all_points)} embeddings to Qdrant for {source_filename}")

        return {"status": "success", "points_saved": len(all_points)}

    except Exception as e:
        logging.error(f"Error saving embeddings to Qdrant for document {doctag}, file {source_filename}: {e}")
        raise


def train_on_document(text, profile_tag, doc_tag, doc_name):
    """Trains and stores embeddings from a document using chunking."""
    try:
        logging.info(f"Starting training for file {doc_name} (document {doc_tag}) in profile {profile_tag}")

        if isinstance(text, dict):
            # Already has embeddings
            result = save_embeddings_to_qdrant(text, profile_tag, doc_tag, doc_name)
            return f"Training complete for {doc_name}. Stored {result.get('points_saved', 0)} embeddings in {profile_tag}"

        elif isinstance(text, str):
            if not text.strip():
                logging.error(f"Empty text content for file {doc_name}")
                raise ValueError(f"Empty content in {doc_name}")

            # Split into chunks
            chunks = text.split(". ")
            chunks = [c.strip() for c in chunks if c.strip()]

            if not chunks:
                logging.error(f"No valid chunks created for file {doc_name}")
                raise ValueError(f"No valid text chunks in {doc_name}")

            logging.info(f"Training on {len(chunks)} chunks for file {doc_name}")
            embeddings = MODEL.encode(chunks, convert_to_numpy=True)

            result = save_embeddings_to_qdrant(
                {"embeddings": embeddings, "texts": chunks},
                profile_tag,
                doc_tag,
                doc_name
            )
            return f"Training complete for {doc_name}. Stored {result.get('points_saved', 0)} embeddings in {profile_tag}"
        else:
            logging.error(f"Unsupported document format for {doc_name}: {type(text)}")
            raise ValueError(f"Unsupported format for {doc_name}: {type(text)}")

    except Exception as e:
        logging.error(f"Error during training for file {doc_name} (document {doc_tag}): {e}")
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

        # Filter only UNDER_REVIEW documents
        under_review_docs = {
            doc_id: doc_info
            for doc_id, doc_info in docColl.items()
            if doc_info['dataDict'].get('status') == 'UNDER_REVIEW'
        }

        logging.info(f"Found {len(under_review_docs)} documents with status UNDER_REVIEW")

        if not under_review_docs:
            logging.warning("No documents with UNDER_REVIEW status")
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
                extracted_doc = doc_data['extractedDoc']
                doc_name = doc_data.get('docName', 'Unknown')

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









