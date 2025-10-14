import json
import uuid
import nltk
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
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize, word_tokenize
from qdrant_client.models import Distance, VectorParams
from api.documentVetting import vettingProcessor
from api.dw_document_extractor import DocumentExtractor
from azure.storage.blob import BlobServiceClient

nltk.download('punkt')

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
docEx = DocumentExtractor()
MODEL = SentenceTransformer(Config.Model.SENTENCE_TRANSFORMERS)
mongoClient = MongoClient(Config.MongoDB.URI)
db = mongoClient[Config.MongoDB.DB]
qdrant_client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API,timeout=60)


def decrypt_data(encrypted_value: str, encryption_key= Config.Encryption.ENCRYPTION_KEY ) -> str:
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
    extracted_data = {}
    try:
        file = file.split('/')[-1]
        if content:
            logging.info(f"Extracting File {file}")
            if file.endswith(".csv"):
                df = pd.read_csv(BytesIO(content))
                extracted_data[file] = docEx.extract_dataframe(df, MODEL)
            elif file.endswith(".xlsx") or file.endswith(".xls"):
                df = pd.read_excel(BytesIO(content))
                extracted_data[file] = docEx.extract_dataframe(df, MODEL)
            elif file.endswith(".json"):
                extracted_data[file] = json.loads(content.decode("utf-8"))
            elif file.endswith(".pdf"):
                extracted_data[file] = docEx.extract_text_from_pdf(content)
            elif file.endswith(".docx"):
                extracted_data[file] = docEx.extract_text_from_docx(content)
            elif file.endswith(".pptx") or file.endswith(".ppt"):
                extracted_data[file] = docEx.extract_text_from_pptx(content)
            elif file.endswith(".txt"):
                extracted_data[file] = content.decode("utf-8")
            else:
                extracted_data[file] = content.decode("utf-8")
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
        subprocess.run(
            ["aws", "configure", "set", "aws_access_key_id", AWS_ACCESS_KEY, "--profile", awsProfile]
        )
        subprocess.run(
            ["aws", "configure", "set", "aws_secret_access_key", AWS_SECRET_KEY, "--profile", awsProfile]
        )
        subprocess.run(
            ["aws", "configure", "set", "region", Region, "--profile", awsProfile]
        )

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
        content = read_s3_file(s3,bucket_name,object_key)
        return content

    except Exception as e:
        return {"Error": str(e)}

def updateVetting(document_id, new_value):
    """
    Updates a specific field in a MongoDB document.

    Args:
        document_id (str): The ID of the document to update.
        field_name (str): The field to update.
        new_value: The new value to set for the field.

    Returns:
        dict: The result of the update operation.
    """
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


def get_azure_docs(files):
    """Fetches documents from Azure Blob Storage."""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(Config.DocAzureBlob.AZURE_BLOB_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(Config.DocAzureBlob.AZURE_BLOB_CONTAINER_NAME)
        # fileName = files.split('/')[-1]
        blob_client = container_client.get_blob_client('local/'+files)
        blob_data = blob_client.download_blob().readall()
        return blob_data

    except Exception as e:
        logging.error(f"Error fetching Azure documents: {e}")
        return []

def connectData(documentConnection):
    dataDict = {}
    for k, v in documentConnection.items():
        docData = v['dataDict']
        connData = v['connDict']
        profileId = docData['profile'].__str__()
        if docData['status'] == 'UNDER_REVIEW':
            try:
                if docData['type'] == 'S3':
                    bkName = connData['s3_details']['bucketName']
                    region = connData['s3_details']['region']
                    ak = decrypt_data(connData['s3_details']['accessKey']).split('\x0c')[0].strip()
                    sk = decrypt_data(connData['s3_details']['secretKey']).split('\x08')[0].strip()
                    s3 = get_s3_client(ak, sk, region)
                    objs = s3.list_objects_v2(Bucket=bkName)
                    file = [obj['Key'] for obj in objs.get("Contents", []) if obj['Key'] == docData['name']]
                    docContent = read_s3_file(s3, bkName, file[0])
                    extractedDoc = fileProcessor(docContent, file[0])
                    vettingPoints = vettingProcessor(extractedDoc)
                    docId = docData['_id'].__str__()
                    profileData = {profileId: extractedDoc}
                    dataDict[docId] = profileData
                    updateVetting(docId, vettingPoints)

                elif docData['type'] == 'LOCAL':
                    files = connData['locations']
                    if len(files) == 1:
                        file = files[0].split('/', 4)[-1]
                        docContent = get_azure_docs(file)
                        extractedDoc = fileProcessor(docContent, file)
                        vettingPoints = vettingProcessor(extractedDoc)
                        docId = docData['_id'].__str__()
                        profileData = {profileId: extractedDoc}
                        dataDict[docId] = profileData
                        updateVetting(docId,vettingPoints)
                    elif len(files) > 1:
                        for file in files:
                            docContent = get_azure_docs(file)
                            extractedDoc = fileProcessor(docContent, file)
                            vettingPoints = vettingProcessor(extractedDoc)
                            docId = docData['_id'].__str__()
                            profileData = {profileId: extractedDoc}
                            dataDict[docId] = profileData
                            updateVetting(docId, vettingPoints)
                    elif len(files) == 0:
                        logging.info("location Empty")
                elif docData['type'] == 'FTP':
                    pass
                elif docData['type'] == 'BLOB':
                    pass
            except Exception as e:
                logging.error(f"Error processing document {docData['name']}: {e}")
        elif docData['status'] == 'DELETED':
            docId = docData['_id'].__str__()
            profileData = docData['profile'].__str__()
            # dataDict[docId] = {profileData: 'Document Deleted'}
            delete_embeddings(profileData, docId)
        else:
            logging.info(v)

    return dataDict

def collectionConnect(name):
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
                docInfo[docId] = {'dataDict':docData,'connDict':connInfo[connRef]}
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
                {"key": "tag", "match": {"value": tag}}
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


def save_embeddings_to_qdrant(embeddings, tags, doctag, batch_size=100):
    """Ensures the collection exists and saves embeddings to Qdrant."""
    try:
        if "embeddings" not in embeddings or embeddings["embeddings"] is None:
            logging.error("Error: 'embeddings' key missing or None!")
            return

        embedding_array = np.asarray(embeddings["embeddings"], dtype=np.float32)
        if embedding_array.size == 0 or embedding_array.shape[1] == 0:
            logging.error("Error: Trying to save empty embeddings!")
            return

        vector_size = embedding_array.shape[1]
        ensure_qdrant_collection(tags, vector_size)

        logging.info(f"Saving embeddings to Qdrant for tag: {tags}")

        all_points = []
        for vector, text in zip(embedding_array, embeddings["texts"]):
            if np.all(vector == 0) or np.size(vector) == 0:
                logging.warning(f"Skipping empty or invalid vector for text: {text}")
                continue

            vector = vector.tolist()

            all_points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"tag": str(tags), "text": text}
                )
            )

        for i in range(0, len(all_points), batch_size):
            batch = all_points[i:i + batch_size]
            qdrant_client.upsert(collection_name=tags, points=batch)
            logging.info(f"Uploaded batch {i // batch_size + 1}/{len(all_points) // batch_size + 1}")

        logging.info("Embeddings successfully saved to Qdrant.")
        trainingStatus = {
            "$set": {
                "tag": doctag,
                "status": 'TRAINING_COMPLETED'
            }
        }
        docdb = db[Config.MongoDB.DOCUMENTS]
        docdb.update_one({"_id": ObjectId(doctag)}, trainingStatus, upsert=True)
        logging.info("Training status updated in Mongo DB")
    except Exception as e:
        logging.error(f"Error saving embeddings to Qdrant: {e}")


def chunk_text(text, chunk_size=256, overlap=128):
    """
    Splits text into overlapping chunks to improve retrieval accuracy.

    Parameters:
    - text (str): The document text.
    - chunk_size (int): Target number of words per chunk.
    - overlap (int): Overlapping words between consecutive chunks.

    Returns:
    - List of chunked text segments.
    """
    sentences = sent_tokenize(text)
    words = [word_tokenize(sent) for sent in sentences]
    words_flat = [word for sent in words for word in sent]  # Flatten word list
    logging.info("Converting to Chunks")
    chunks = []
    for i in range(0, len(words_flat), chunk_size - overlap):
        chunk = words_flat[i:i + chunk_size]
        chunks.append(" ".join(chunk))

    return chunks

def train_on_document(text, profile_tag,doc_tag):
    """Trains and stores embeddings from a document using chunking."""
    try:
        if isinstance(text, dict):
            save_embeddings_to_qdrant(text,profile_tag,doc_tag)
        elif isinstance(text, str):
            chunks = chunk_text(text)
            logging.info(f"Training on document with tag: {profile_tag}")
            embeddings = MODEL.encode(chunks, convert_to_numpy=True)
            save_embeddings_to_qdrant({"embeddings": embeddings, "texts": chunks},profile_tag,doc_tag)
        else:
            logging.error("Unsupported document format for training.")
            return "Training failed."
        return f"Training complete. Model stored as {profile_tag}!"
    except Exception as e:
        logging.error(f"Error during training: {e}")
        return f"Training failed for tag {profile_tag}"


def trainData():
    try:
        logging.info(f"Starting training")
        docColl = extract_document_info()
        resData = connectData(docColl)
        for doc_id,profileData in resData.items():
            for profile_id, docText in profileData.items():
                for docName, docContent in docText.items():
                    docTrain = train_on_document(docContent,profile_id,doc_id)
                    logging.info(docTrain)
    except Exception as e:
        logging.error(f"Error in training data: {e}")
        return "Training failed."

# trainData()