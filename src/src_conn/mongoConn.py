from pymongo import MongoClient
import boto3 as b3
from Crypto.Cipher import AES
import hashlib
import pandas as pd
import json
import fitz
from typing import List
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.vectorstores import VectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import Config


mongoUri = 'mongodb+srv://admin:nj4pJfO3e1FcGXDo@cluster0.47nz8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
client = MongoClient(mongoUri)
#print(client.list_database_names())  #['test', 'admin', 'local']
db = client["test"]
# coll = db.list_collection_names() #['customerlayouts', 'layouts', 'connectors', 'customers', 'cases', 'documents', 'users', 'subscriptions', 'profiles', 'casehistories', 'companyproducts', 'customerProd', 'customerproducts']



def get_s3_client(AWS_ACCESS_KEY,AWS_SECRET_KEY,Region):
    return b3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=Region,
    )

def decrypt_data(encrypted_value: str, encryption_key: str) -> str:
    # Generate the key from the encryption key
    key = hashlib.scrypt(encryption_key.encode(), salt=b'salt', n=16384, r=8, p=1, dklen=32)

    # Split IV and Encrypted Data
    iv_hex, encrypted = encrypted_value.split(':')
    iv = bytes.fromhex(iv_hex)
    encrypted_bytes = bytes.fromhex(encrypted)

    # Initialize AES Cipher (AES-256-CBC)
    cipher = AES.new(key, AES.MODE_CBC, iv)

    # Decrypt and remove padding
    decrypted = cipher.decrypt(encrypted_bytes)
    decrypted_text = decrypted.rstrip(b"\x00").decode('utf-8')

    return decrypted_text


def read_s3_file(bucket, file_key):
    obj = s3b.get_object(Bucket=bucket, Key=file_key)
    content = obj["Body"].read()
    return content


def extract_text_from_pdf(pdf_bytes):
    text = ""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text


class Ingestor:
    def __init__(self):
        self.embeddings = FastEmbedEmbeddings(model_name=Config.Model.EMBEDDINGS)
        self.semantic_splitter = SemanticChunker(
            self.embeddings, breakpoint_threshold_type="interquartile"
        )
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=128,
            add_start_index=True,
        )

    def ingest_extracted(self, extracted_contents: List[str]) -> VectorStore:
        """
        Ingest extracted text content directly instead of file paths.
        """
        documents = []
        for content in extracted_contents:
            documents.extend(
                self.recursive_splitter.split_documents(
                    self.semantic_splitter.create_documents([content])
                )
            )

        return Qdrant.from_documents(
            documents=documents,
            embedding=self.embeddings,
            url=Config.Qdrant.URL,
            api_key = Config.Qdrant.API,
            collection_name=Config.Database.DOCUMENTS_COLLECTION,
        )


def extractData(s3):
    objs = s3.list_objects_v2(Bucket=testConn['bucketName'])
    file_list = [obj["Key"] for obj in objs["Contents"]]
    if file_list:
        for file in file_list:
            try:
                content = read_s3_file(testConn['bucketName'], file)

                # Auto detect content type and display accordingly
                if file.endswith(".csv"):
                    df = pd.read_csv(content)
                    # pd.Dataframe(df)
                elif file.endswith(".json"):
                    parsed_json = json.loads(content.decode("utf-8"))
                    return parsed_json
                elif file.endswith(".pdf"):
                    extracted_text = extract_text_from_pdf(content)
                    return extracted_text
                else:
                    return content.decode("utf-8")

            except Exception as e:
                print(f"Error reading file {file}: {e}")
    else:
        print("No files found in the selected bucket.")



def connStr():
    conColl = db['connectors']
    connectors = conColl.find()
    s3conn = {}
    for con in connectors:
        if con['type'] == 'S3':
            s3conn[con['name']] = con['s3_details']
    return s3conn


conn = connStr()
for k,v in conn.items():
    decryptedAK =  decrypt_data(v['accessKey'],'J9cuHrESAz')
    decryptedSK =  decrypt_data(v['secretKey'],'J9cuHrESAz')
    v['accessKey'] = decryptedAK.split('\x0c')[0].strip()
    v['secretKey'] = decryptedSK.split('\x08')[0].strip()
    conn[k] = v

testConn = conn['S3 test connection']
# s3b = get_s3_client(testConn['accessKey'],testConn['secretKey'],testConn['region'])
# fileData = extractData(s3b)
# trainingVector = Ingestor().ingest_extracted([fileData])
# print(trainingVector)
