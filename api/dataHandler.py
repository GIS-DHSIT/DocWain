import boto3 as b3
import json
import fitz
import docx
import logging
from Crypto.Cipher import AES
import hashlib
from pymongo import MongoClient
import pandas as pd
import numpy as np
from bson.objectid import ObjectId
from sentence_transformers import SentenceTransformer
from api.config import Config
from io import BytesIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL = SentenceTransformer(Config.Model.SENTENCE_TRANSFORMERS)
mongoClient = MongoClient(Config.MongoDB.URI)
db = mongoClient[Config.MongoDB.DB]
chunks_collection = db["embeddings_chunks"]  # New collection for chunked embeddings

def chunk_embeddings(embeddings, chunk_size=500):
    """Splits a large embeddings array into smaller chunks."""
    return [embeddings[i:i + chunk_size] for i in range(0, len(embeddings), chunk_size)]

def save_embeddings_to_mongo(db_collection, embeddings, tags):
    """Saves large embeddings in chunks to avoid exceeding MongoDB's 16MB limit."""
    try:
        logging.info(f"Saving embeddings to MongoDB for profile ID: {tags}")

        # Split large embeddings into chunks
        chunk_ids = []
        for chunk in chunk_embeddings(embeddings["embeddings"]):
            chunk_doc = {"profile": tags, "chunk": chunk.tolist()}
            chunk_id = chunks_collection.insert_one(chunk_doc).inserted_id
            chunk_ids.append(chunk_id)

        embedding_data = {
            "$set": {
                "tag": tags,
                "embedding_chunks": chunk_ids,  # Store references instead of full embeddings
                "texts": embeddings["texts"],
                "status": 'TRAINING_COMPLETED'
            }
        }

        db_collection.update_one({"_id": ObjectId(tags)}, embedding_data, upsert=True)
        logging.info("Embeddings successfully saved in chunks.")
    except Exception as e:
        logging.error(f"Error saving embeddings to MongoDB: {e}")

def load_embeddings_from_chunks(chunk_ids):
    """Loads and reconstructs large embeddings from stored chunks."""
    chunks = []
    for chunk_id in chunk_ids:
        chunk_doc = chunks_collection.find_one({"_id": chunk_id})
        if chunk_doc:
            chunks.extend(chunk_doc["chunk"])
    return np.array(chunks, dtype=np.float32)  # Convert back to NumPy array

def train_on_document(text, db_collection, profiletag, docTag):
    """Trains and stores embeddings from a document using chunking."""
    try:
        logging.info(f"Training on document with tag: {profiletag}")
        chunks = text.split(". ")
        embeddings = MODEL.encode(chunks, convert_to_numpy=True)
        save_embeddings_to_mongo(db_collection, {"embeddings": embeddings, "texts": chunks}, docTag)
        return f"Training complete. Model stored as {profiletag}!"
    except Exception as e:
        logging.error(f"Error during training: {e}")
        return f"Training failed for tag {profiletag}"

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
    """Extracts data from all files stored in an S3 bucket."""
    extracted_data = {}  # Dictionary to store file content

    try:
        logging.info(f"Extracting data from S3 bucket: {bucket}")
        objs = s3.list_objects_v2(Bucket=bucket)
        file_list = [obj["Key"] for obj in objs.get("Contents", [])]

        if not file_list:
            logging.warning("No files found in the selected bucket.")
            return None

        for file in file_list:
            try:
                content = read_s3_file(s3, bucket, file)
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
            except Exception as e:
                logging.error(f"Error processing file {file}: {e}")

    except Exception as e:
        logging.error(f"Error extracting data from bucket {bucket}: {e}")
        return None

    return extracted_data  # Return dictionary of file data

def connStr(dbName='connectors'):
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
        getProfiles = documents.collection.distinct('profile')
        profiles = {}
        for docs in documents:
            if docs['type'] == 'S3' and docs['profile'] in getProfiles:
                # if docs['status'] == 'TRAINING_PENDING':
                profName = docs['profile'].__str__()
                docName = docs['_id'].__str__()
                profiles[profName + '_' + docName] = docs
            elif docs['type'] == 'LOCAL':
                print(docs) #TODO update the code for LOCAL files
        return {'profiles': profiles, 'db': docColl}
    except Exception as e:
        logging.error(f"Error fetching document details: {e}")
        return {}

def initiate_training(collections):
    """Initiates training on extracted documents."""
    try:
        logging.info("Starting training process.")
        for uid, details in collections.items():
            if 'region' in details:
                tag = details['profile'].__str__()
                s3b = get_s3_client(details['accessKey'], details['secretKey'], details['region'])
                bucketName = details['bucketName']
                fileData = extractData(s3b, bucketName)
                return {'fileData': fileData, 'tag': tag}
            elif 'type' == 'LOCAL':
                print(details)#TODO update the code
            else:
                logging.error("No Region details found")
        logging.warning("No valid collections found for training.")
        return 'Not Found'
    except Exception as e:
        logging.error(f"Error in training initiation: {e}")


def extractCont(documentData,collectionName,schemaName):
    """Extracts content details from document metadata."""
    try:
        logging.info("Extracting content from document data.")
        reqDict = {}
        connData = connStr(collectionName) #'connectors'
        connDet = connData[schemaName] #'actual documents provided '

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
                logging.error(f"Error decrypting {fileDetails}: {e}")
        resDict = {}
        for id, data in documentData.items():
            resDict[id]= data | reqDict

        logging.info("Content extraction complete.")
        return resDict
    except Exception as e:
        logging.error(f"Critical failure in extractCont: {e}")
        return []

def trainData(collectionDir,schemaName, collectionName='documents'):
    """Orchestrates the full training pipeline."""
    try:
        logging.info(f"Starting training for collection: {collectionName}")
        documentData = docStr(collectionName)
        coll = extractCont(documentData['profiles'],collectionDir, schemaName)
        data = initiate_training(coll)
        if data != 'Not Found':
            combined_dict = {}
            for key1, value1 in data['fileData'].items():
                for key2, nested_dict in coll.items():
                    if nested_dict.get('location') == key1:  # Matching based on 'name'
                        combined_dict[nested_dict['_id']] = {'id': key2, 'text': value1, **nested_dict}
                        break
            for objIds, Data in combined_dict.items():
                status = train_on_document(Data['text'], documentData['db'], Data['profile'], objIds)
                logging.info("Training completed.")
                logging.info(status)
            statusCon = db[collectionName]
            op = pd.DataFrame(statusCon.find())
            status_response = op[['name', 'profile', 'status']]

            return status_response.to_json(orient='records')
        else:
            return "No data to be processed!"

    except Exception as e:
        logging.error(f"Error in training data: {e}")
        return "Training failed."

# trainData("connectors","actual documents provided ","documents")