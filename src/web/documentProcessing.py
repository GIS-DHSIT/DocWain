from pymongo import MongoClient
import boto3 as b3
from botocore.config import Config
from Crypto.Cipher import AES
import hashlib
import pandas as pd
import json
import fitz
import asyncio
from typing import List
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.vectorstores import VectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.web.webConfig import Config
from src.web.model import create_llm
from src.web.retriever import create_retriever
from src.web.chain import create_chain,ask_question
# from src.web.ingestor import Ingestor


mongoUri = Config.MongoDB.URI
client = MongoClient(mongoUri)
#print(client.list_database_names())  #['test', 'admin', 'local']
db = client["test"]
# coll = db.list_collection_names() #['customerlayouts', 'layouts', 'connectors', 'customers', 'cases', 'documents', 'users', 'subscriptions', 'profiles', 'casehistories', 'companyproducts', 'customerProd', 'customerproducts']


def get_session_token(aws_access_key_id, aws_secret_access_key, duration_seconds=3600):
    """
    Get a session token using provided AWS Access Key ID and Secret Access Key.

    :param aws_access_key_id: Your AWS Access Key ID
    :param aws_secret_access_key: Your AWS Secret Access Key
    :param duration_seconds: Duration of the session in seconds (default: 1 hour)
    :return: Temporary credentials (AccessKeyId, SecretAccessKey, SessionToken)
    """
    try:
        sts_client = b3.client(
            "sts",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        response = sts_client.get_session_token(DurationSeconds=duration_seconds)
        credentials = response["Credentials"]
        return credentials

    except Exception as e:
        print(f"Error getting session token: {e}")
        return None

def get_s3_client(AWS_ACCESS_KEY,AWS_SECRET_KEY,Region):
    sessionTkn = get_session_token(AWS_ACCESS_KEY,AWS_SECRET_KEY)
    return b3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        # aws_session_token = sessionTkn['SessionToken'],
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


def read_s3_file(s3,bucket, file_key):
    obj = s3.get_object(Bucket=bucket, Key=file_key)
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

    def ingest_extracted(self, uid, extracted_contents: List[str]) -> VectorStore:
        """
        Ingest extracted text content directly instead of file paths.
        Each document is assigned a unique ID for easy reference.
        """
        documents = []
        for content in extracted_contents:
            split_docs = self.recursive_splitter.split_documents(
                self.semantic_splitter.create_documents([content])
            )

            # Assign unique IDs to each document
            for doc in split_docs:
                doc.metadata["id"] = uid# Add unique ID

            documents.extend(split_docs)

        return Qdrant.from_documents(
            documents=documents,
            embedding=self.embeddings,
            url=Config.Qdrant.URL,
            api_key=Config.Qdrant.API,
            collection_name=Config.Database.DOCUMENTS_COLLECTION,
        )


def extractData(s3, bucket):
    objs = s3.list_objects_v2(Bucket=bucket)
    file_list = [obj["Key"] for obj in objs["Contents"]]
    if file_list:
        for file in file_list:
            try:
                content = read_s3_file(s3,bucket, file)

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



def connStr(dbName):
    conColl = db[dbName]
    connectors = conColl.find()
    conn = {}
    for con in connectors:
        if con['type'] == 'S3':
            dconn = con['s3_details']
            # dconn[con['_id'].__str__()] = con['_id']
            conn[con['name']] = dconn
        elif con['type'] == 'LOCAL':
            for items in con['files']:
                for locs in con['locations']:
                    if items['name'] in locs:
                        items['locations'] = locs
                conn[con['name']+'_'+items['name']] = items
    return conn

def docStr(dbName):
    docColl = db[dbName]
    documents = docColl.find()
    getProfiles = documents.collection.distinct('profile')
    profiles = {}
    for docs in documents:
        if docs['type'] == 'S3':
            profName = docs['profile'].__str__()
            docName = docs['_id'].__str__()
            profiles[profName+'_'+docName] = docs
    return profiles

def updateMongo(dbName, filterVal,vecUrl):
    docColl = db[dbName]
    filter_criteria = {"profile": filterVal}
    update_data = {"$set": {"VectorUrl": vecUrl,"status":'TRAINING_COMPLETED'}}
    res = docColl.update_one(filter_criteria,update_data)
    if res.matched_count > 0:
        print(f"Successfully updated {res.modified_count} document(s). New field added.")
    else:
        print("No document found with the specified criteria.")

async def ask_chain(question: str, chain,tag):
    documents = []
    full_response = ""
    async for event in ask_question(tag,chain,question,session_id="session-1"):
        if type(event) is str:
            full_response += event
        if type(event) is list:
            documents.extend(event)
        for i, doc in enumerate(documents):
            print(doc.page_content)


def initiate_training(collections):
    for items in collections:

        if 'region' in items:
            tag = items['profile'].__str__()
            # print(items)
            s3b = get_s3_client(items['accessKey'],items['secretKey'],items['region'])
            bucketName = items['bucketName']
            fileData = extractData(s3b, bucketName)
            trainingVector = Ingestor().ingest_extracted(tag,[fileData])
            # val = [val for val in list(connectionData.values())if type(val)!=str][0]
            updateMongo('documents',items['profile'],Config.Qdrant.URL)
            return {'trainingVector':trainingVector, 'tag':tag}
        else:
            pass #TODO add a method to handle local uploads

def retrieveTraining(trainInp):
    trainingVec = trainInp['trainingVector']
    tag = trainInp['tag']
    llm = create_llm()
    retriever = create_retriever(llm, vector_store=trainingVec)
    ret = create_chain(tag,llm, retriever)
    # op = ask_question(,ret,"summarize the document",'1')
    asyncio.run(ask_chain("summarize the document", ret,tag))


def extractCont(documentData):
    colls = []
    for k,v in documentData.items():
        floc = v['location'].split('/')
        dbNames = v['location'].split('/')[0].strip()
        connExtract = connStr(dbNames)
        # print(connExtract)
        for connName,connDet in connExtract.items():
            reqDict = {}
            for fileDetails,vals in connDet.items():
                if fileDetails == 'accessKey':
                    decryptedAK =  decrypt_data(vals,'J9cuHrESAz')#TODO Needs to be parameterized
                    reqDict[fileDetails] = decryptedAK.split('\x0c')[0].strip()
                elif fileDetails == 'secretKey':
                    decryptedSK = decrypt_data(vals, 'J9cuHrESAz')#TODO Needs to be parameterized
                    reqDict[fileDetails] = decryptedSK.split('\x08')[0].strip()
                else:
                    reqDict[fileDetails] = vals
            reqDict['name'] = connName
            reqDict['profile'] = documentData[k]['profile']
            reqDict['uid'] = k
            reqDict['docName'] = documentData[k]['name']
            reqDict['location'] = floc[1]
            colls.append(reqDict)
    return colls

documentData = docStr('documents')
coll = extractCont(documentData)
trained = initiate_training(coll)
trainOp = retrieveTraining(trained)

# def handleConnection(dbName, connName):
#     conn = connStr(dbName)
#     connData = conn[connName]
#     connKeys = {}
#     for k,v in connData.items():
#         if k == 'accessKey':
#             decryptedAK =  decrypt_data(connData['accessKey'],'J9cuHrESAz')
#             connKeys[k] = decryptedAK.split('\x0c')[0].strip()
#
#         elif k == 'secretKey':
#             decryptedSK = decrypt_data(connData['secretKey'], 'J9cuHrESAz')
#             connKeys[k] = decryptedSK.split('\x08')[0].strip()
#         else:
#             connKeys[k] =v
#     return connKeys

# connectionData = handleConnection('connectors','S3 test connection')