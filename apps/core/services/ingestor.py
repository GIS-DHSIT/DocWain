from typing import List, Optional
from django.core.files.uploadedfile import UploadedFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    CSVLoader
)
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import tempfile
from pathlib import Path
from django.conf import settings
from apps.core.utils.logger import setup_logger
logger = setup_logger()

class Ingestor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=128
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5"
        )

    async def ingest(self, files: List[UploadedFile]) -> FAISS:
        """Process and ingest documents into vector store"""
        try:
            documents = []

            for file in files:
                # Save to temporary file
                suffix = Path(file.name).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    for chunk in file.chunks():
                        tmp.write(chunk)
                    tmp_path = Path(tmp.name)

                try:
                    # Process based on file type
                    if suffix == '.pdf':
                        loader = PyPDFLoader(str(tmp_path))
                    elif suffix in ['.docx', '.doc']:
                        loader = Docx2txtLoader(str(tmp_path))
                    elif suffix in ['.xlsx', '.xls']:
                        loader = UnstructuredExcelLoader(str(tmp_path))
                    elif suffix == '.csv':
                        loader = CSVLoader(str(tmp_path))
                    else:
                        raise ValueError(f"Unsupported file type: {suffix}")

                    documents.extend(loader.load())

                finally:
                    # Cleanup temp file
                    tmp_path.unlink(missing_ok=True)

            # Split documents
            chunks = self.text_splitter.split_documents(documents)

            # Create vector store
            vector_store = FAISS.from_documents(chunks, self.embeddings)

            # Save to disk if configured
            if hasattr(settings, 'VECTOR_STORE_PATH'):
                vector_store.save_local(settings.VECTOR_STORE_PATH)

            return vector_store

        except Exception as e:
            logger.error(f"Error during document ingestion: {e}")
            raise
