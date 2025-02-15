import boto3 as b3
import json
import fitz
import docx
import logging
from Crypto.Cipher import AES
import hashlib
from pymongo import MongoClient
import pandas as pd
from bson.objectid import ObjectId
from sentence_transformers import SentenceTransformer
from api.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL = SentenceTransformer(Config.Model.SENTENCE_TRANSFORMERS)
mongoClient = MongoClient(Config.MongoDB.URI)
db = mongoClient[Config.MongoDB.DB]

def save_embeddings_to_mongo(db_collection, embeddings, tags):
    """Saves FAISS index and text data to MongoDB."""
    try:
        logging.info(f"Saving embeddings to MongoDB for profile ID: {tags}")
        embedding_data = {"$set": {
            "tag": tags,
            "embeddings": embeddings["embeddings"].tolist(),
            "texts": embeddings["texts"],
            "status": 'TRAINING_COMPLETED'
        }}
        db_collection.update_one({"profile": ObjectId(tags)}, embedding_data, upsert=True)
        logging.info("Embeddings successfully saved.")
    except Exception as e:
        logging.error(f"Error saving embeddings to MongoDB: {e}")

def train_on_document(text, db_collection, tag="default"):
    """Trains and stores embeddings from a document."""
    try:
        logging.info(f"Training on document with tag: {tag}")
        chunks = text.split(". ")
        embeddings = MODEL.encode(chunks, convert_to_numpy=True)
        save_embeddings_to_mongo(db_collection, {"embeddings": embeddings, "texts": chunks}, tag)
        return f"Training complete. Model stored as {tag}!"
    except Exception as e:
        logging.error(f"Error during training: {e}")
        return f"Training failed for tag {tag}"

def decrypt_data(encrypted_value: str, encryption_key: str) -> str:
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
        logging.info(f"Extracting text from PDF: {pdf_path}")
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

def extractData(s3, bucket):
    """Extracts data from files stored in an S3 bucket."""
    try:
        logging.info(f"Extracting data from S3 bucket: {bucket}")
        objs = s3.list_objects_v2(Bucket=bucket)
        file_list = [obj["Key"] for obj in objs.get("Contents", [])]

        if file_list:
            for file in file_list:
                try:
                    content = read_s3_file(s3, bucket, file)
                    if content:
                        if file.endswith(".csv"):
                            return pd.read_csv(content)
                        elif file.endswith(".json"):
                            return json.loads(content.decode("utf-8"))
                        elif file.endswith(".pdf"):
                            return extract_text_from_pdf(content)
                        elif file.endswith(".docx"):
                            return extract_text_from_docx(content)
                        else:
                            return content.decode("utf-8")
                except Exception as e:
                    logging.error(f"Error processing file {file}: {e}")
        else:
            logging.warning("No files found in the selected bucket.")
    except Exception as e:
        logging.error(f"Error extracting data from bucket {bucket}: {e}")

def connStr(dbName):
    """Retrieves connector details from MongoDB."""
    try:
        logging.info(f"Fetching connection details for database: {dbName}")
        conColl = db[dbName]
        connectors = conColl.find()
        conn = {}

        for con in connectors:
            if con['type'] == 'S3':
                conn[con['name']] = con['s3_details']
            elif con['type'] == 'LOCAL':
                for items in con['files']:
                    conn[con['name']+'_'+items['name']] = items
        return conn
    except Exception as e:
        logging.error(f"Error fetching connection details: {e}")
        return {}

def docStr(dbName):
    """Retrieves document metadata from MongoDB."""
    try:
        logging.info(f"Fetching document details for collection: {dbName}")
        docColl = db[dbName]
        documents = docColl.find()
        profiles = {docs['profile'].__str__() + '_' + docs['_id'].__str__(): docs for docs in documents}
        return {'profiles': profiles, 'db': docColl}
    except Exception as e:
        logging.error(f"Error fetching document details: {e}")
        return {}

def initiate_training(collections):
    """Initiates training on extracted documents."""
    try:
        logging.info("Starting training process.")
        for items in collections:
            if 'region' in items:
                tag = items['profile'].__str__()
                s3b = get_s3_client(items['accessKey'], items['secretKey'], items['region'])
                bucketName = items['bucketName']
                fileData = extractData(s3b, bucketName)
                return {'fileData': fileData, 'tag': tag}
        logging.warning("No valid collections found for training.")
        return 'Not Found'
    except Exception as e:
        logging.error(f"Error in training initiation: {e}")


def extractCont(documentData):
    """Extracts content details from document metadata."""
    try:
        logging.info("Extracting content from document data.")
        colls = []

        for k, v in documentData.items():
            try:
                floc = v['location'].split('/')
                dbNames = v['location'].split('/')[0].strip()
                connExtract = connStr(dbNames)
                logging.info(f"Processing document: {v['name']} with profile: {v['profile']}")

                for connName, connDet in connExtract.items():
                    reqDict = {}
                    for fileDetails, vals in connDet.items():
                        try:
                            if fileDetails == 'accessKey':
                                decryptedAK = decrypt_data(vals, 'J9cuHrESAz')  # TODO Needs to be parameterized
                                reqDict[fileDetails] = decryptedAK.split('\x0c')[0].strip()
                            elif fileDetails == 'secretKey':
                                decryptedSK = decrypt_data(vals, 'J9cuHrESAz')  # TODO Needs to be parameterized
                                reqDict[fileDetails] = decryptedSK.split('\x08')[0].strip()
                            else:
                                reqDict[fileDetails] = vals
                        except Exception as e:
                            logging.error(f"Error decrypting {fileDetails} for connection {connName}: {e}")

                    reqDict['name'] = connName
                    reqDict['profile'] = documentData[k]['profile']
                    reqDict['uid'] = k
                    reqDict['docName'] = documentData[k]['name']
                    reqDict['location'] = floc[1]

                    colls.append(reqDict)
            except Exception as e:
                logging.error(f"Error processing document ID {k}: {e}")

        logging.info("Content extraction complete.")
        return colls
    except Exception as e:
        logging.error(f"Critical failure in extractCont: {e}")
        return []

def trainData(collectionName='documents'):
    """Orchestrates the full training pipeline."""
    try:
        logging.info(f"Starting training for collection: {collectionName}")
        documentData = docStr(collectionName)
        coll = extractCont(documentData['profiles'])
        data = initiate_training(coll)
        if data != 'Not Found':
            status = train_on_document(data['fileData'], documentData['db'], data['tag'])
            return status
        return "Training initiation failed."
    except Exception as e:
        logging.error(f"Error in training data: {e}")
        return "Training failed."