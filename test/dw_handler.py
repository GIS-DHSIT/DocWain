import json
import fitz
import docx
import uuid
import logging
import hashlib
import boto3 as b3
import numpy as np
import pandas as pd
from io import BytesIO
from Crypto.Cipher import AES
from pymongo import MongoClient
from urllib.parse import urlparse
from bson.objectid import ObjectId
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Distance, VectorParams
from src.api.config import Config


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        logging.info(f"Extracting text from PDF")
        text = ""
        with fitz.open(stream=pdf_path, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
        logging.info("Text extraction from PDF successful.")
        return text
    except Exception as e:
        logging.error(f"Failed to extract text from PDF: {e}")
        return ""

def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file."""
    try:
        logging.info(f"Extracting text from DOCX: {docx_path}")
        doc = docx.Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        logging.info("Text extraction from DOCX successful.")
        return text
    except Exception as e:
        logging.error(f"Failed to extract text from DOCX: {e}")
        return ""

def fileProcessor(content,file):
    extracted_data = {}
    try:
        if content:
            logging.info(f"Extracting File {file}")
            if file.endswith(".csv"):
                extracted_data[file] = pd.read_csv(BytesIO(content))
            elif file.endswith(".json"):
                extracted_data[file] = json.loads(content.decode("utf-8"))
            elif file.endswith(".pdf"):
                extracted_data[file] = extract_text_from_pdf(content)
            elif file.endswith(".docx"):
                extracted_data[file] = extract_text_from_docx(content)
            else:
                extracted_data[file] = content.decode("utf-8")
        return extracted_data
    except Exception as e:
        logging.error(f"Error processing file {file}: {e}")


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
        logging.info("Creating S3 client.")
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
    s3 = b3.client('s3')
    try:
        content = read_s3_file(s3,bucket_name,object_key)
        return content

    except Exception as e:
        return {"Error": str(e)}


def connectData(documentConnection):
    dataDict = {}
    for k, v in documentConnection.items():
        docData = v['dataDict']
        connData = v['connDict']
        profileId = docData['profile'].__str__()
        if docData['status'] == 'TRAINING_PENDING':
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
                docId = docData['_id'].__str__()
                profileData = {profileId: extractedDoc}
                dataDict[docId] = profileData
            elif docData['type'] == 'LOCAL':
                files = connData['locations']
                if len(files) ==1:
                    file = files[0]
                    docContent = get_s3_document_info(file)
                    extractedDoc = fileProcessor(docContent, file)
                    docId = docData['_id'].__str__()
                    profileData = {profileId: extractedDoc}
                    dataDict[docId] = profileData
                else:
                    for file in files:
                        docContent = get_s3_document_info(file)
                        extractedDoc = fileProcessor(docContent, file)
                        docId = docData['_id'].__str__()
                        profileData = {profileId: extractedDoc}
                        dataDict[docId] = profileData
            elif docData['type'] == 'FTP':
                pass
            elif docData['type'] == 'BLOB':
                pass
        else:
            print(v)

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

def train_on_document(text, profile_tag,doc_tag):
    """Trains and stores embeddings from a document using chunking."""
    try:
        logging.info(f"Training on document with tag: {profile_tag}")
        chunks = text.split(". ")
        embeddings = MODEL.encode(chunks, convert_to_numpy=True)
        save_embeddings_to_qdrant({"embeddings": embeddings, "texts": chunks},profile_tag,doc_tag)
        return f"Training complete. Model stored as {profile_tag}!"
    except Exception as e:
        logging.error(f"Error during training: {e}")
        return f"Training failed for tag {profile_tag}"

def train():
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


train()
