from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import pandas as pd
import numpy as np
from langchain.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import boto3
from ftplib import FTP
import os
import shutil
import json

app = FastAPI(title="Docwain API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configuration models
class SourceConfig(BaseModel):
    source_type: str
    connection_string: str


class ChatMessage(BaseModel):
    message: str
    chat_history: Optional[List[tuple]] = []


# Document processing class
class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.chat_model = ChatOpenAI(temperature=0.7)

    def process_file(self, file_path: str, file_type: str):
        if file_type == 'pdf':
            loader = PyPDFLoader(file_path)
        elif file_type == 'csv':
            loader = CSVLoader(file_path)
        elif file_type == 'ppt' or file_type == 'pptx':
            loader = UnstructuredPowerPointLoader(file_path)
        elif file_type == 'docx':
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_type == 'xlsx':
            loader = UnstructuredExcelLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        documents = loader.load()
        splits = self.text_splitter.split_documents(documents)

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
        else:
            self.vector_store.add_documents(splits)

    def get_response(self, query: str, chat_history: List[tuple]) -> str:
        if self.vector_store is None:
            raise ValueError("No documents have been processed yet")

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.chat_model,
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True
        )

        result = qa_chain({"question": query, "chat_history": chat_history})
        return result["answer"]


# Source connector class
class SourceConnector:
    @staticmethod
    def connect_s3(connection_string: str):
        try:
            # Parse connection string (format: bucket_name:access_key:secret_key)
            bucket, access_key, secret_key = connection_string.split(':')
            s3 = boto3.client(
                's3',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key
            )
            return s3, bucket
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to connect to S3: {str(e)}")

    @staticmethod
    def connect_ftp(connection_string: str):
        try:
            # Parse connection string (format: host:user:password)
            host, user, password = connection_string.split(':')
            ftp = FTP(host)
            ftp.login(user=user, passwd=password)
            return ftp
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to connect to FTP: {str(e)}")


# Initialize processor
document_processor = DocumentProcessor()


# API endpoints
@app.post("/configure-source")
async def configure_source(config: SourceConfig):
    try:
        if config.source_type == "s3":
            s3, bucket = SourceConnector.connect_s3(config.connection_string)
            # Test connection by listing buckets
            s3.list_buckets()
        elif config.source_type == "ftp":
            ftp = SourceConnector.connect_ftp(config.connection_string)
            # Test connection by listing directory
            ftp.nlst()
        elif config.source_type == "local":
            if not os.path.exists(config.connection_string):
                raise ValueError("Directory does not exist")
        else:
            raise ValueError("Unsupported source type")

        return {"message": "Source configured successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    try:
        # Create temporary file
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the document
        file_type = file.filename.split('.')[-1].lower()
        document_processor.process_file(temp_file_path, file_type)

        # Clean up
        os.remove(temp_file_path)

        return {"message": "Document processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/chat")
async def chat(message: ChatMessage):
    try:
        response = document_processor.get_response(
            message.message,
            message.chat_history
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)