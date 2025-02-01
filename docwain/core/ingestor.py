from typing import List, Callable, Any, Optional, Dict, Union
from pathlib import Path
import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    CSVLoader
)
from langchain_community.vectorstores import FAISS, Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
import os,sys
path = os.getcwd()
sys.path.append(path)
from docwain.config import Config
from docwain.utils.logger import setup_logger
from docwain.utils.helpers import save_uploaded_file, cleanup_temp_files

logger = setup_logger()


def process_document(file_path: Path) -> List[Any]:
    """Process a single document - to be run in a separate process"""
    try:
        suffix = file_path.suffix.lower()
        loaders = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.doc': Docx2txtLoader,
            '.xlsx': UnstructuredExcelLoader,
            '.xls': UnstructuredExcelLoader,
            '.csv': CSVLoader
        }

        if suffix not in loaders:
            raise ValueError(f"Unsupported file type: {suffix}")

        loader_class = loaders[suffix]
        loader = loader_class(str(file_path))
        return loader.load()
    except Exception as e:
        logger.error(f"Error processing document {file_path}: {e}")
        return []


class Ingestor:
    """Handles document ingestion and vectorization with parallel processing"""

    def __init__(self, progress_callback: Optional[Callable[[int, int, str], None]] = None):
        self.progress_callback = progress_callback
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=Config.Model.EMBEDDINGS
        )

        # Set number of processes
        self.num_processes = max(1, mp.cpu_count() - 1)

        # Initialize Qdrant client if configured
        self.use_qdrant = Config.Database.USE_QDRANT
        if self.use_qdrant:
            try:
                self.client = QdrantClient(
                    host=Config.Database.QDRANT_HOST,
                    port=Config.Database.QDRANT_PORT
                )
            except Exception as e:
                logger.warning(f"Failed to connect to Qdrant, falling back to FAISS: {e}")
                self.use_qdrant = False

    async def ingest(self, files: List[Any]) -> Union[Qdrant, FAISS]:
        """Process and ingest documents into vector store using parallel processing"""
        temp_paths = []
        try:
            # Save uploaded files
            temp_paths = await self._save_files(files)

            # Process documents in parallel
            documents = await self._process_documents_parallel(temp_paths)

            # Split into chunks
            if self.progress_callback:
                self.progress_callback(0, 1, "Splitting documents...")
            chunks = self.text_splitter.split_documents(documents)

            # Create vector store
            if self.progress_callback:
                self.progress_callback(0, 1, "Creating vector store...")

            # Try Qdrant first, fall back to FAISS if needed
            try:
                if self.use_qdrant:
                    return await self._create_qdrant_store(chunks)
                else:
                    raise ValueError("Qdrant not configured, using FAISS")
            except Exception as e:
                logger.warning(f"Failed to use Qdrant, falling back to FAISS: {e}")
                return self._create_faiss_store(chunks)

        except Exception as e:
            logger.error(f"Error during document ingestion: {e}")
            raise
        finally:
            # Cleanup temporary files
            await cleanup_temp_files(temp_paths)

    async def _save_files(self, files: List[Any]) -> List[Path]:
        """Save uploaded files with progress tracking"""
        temp_paths = []
        for i, file in enumerate(files, 1):
            if self.progress_callback:
                self.progress_callback(i, len(files), f"Saving {file.name}...")
            temp_path = await save_uploaded_file(file.read(), file.name)
            temp_paths.append(temp_path)
        return temp_paths

    async def _process_documents_parallel(self, file_paths: List[Path]) -> List[Any]:
        """Process documents in parallel using ProcessPoolExecutor"""
        if self.progress_callback:
            self.progress_callback(0, len(file_paths), "Processing documents in parallel...")

        documents = []
        loop = asyncio.get_event_loop()

        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            futures = [
                loop.run_in_executor(executor, process_document, file_path)
                for file_path in file_paths
            ]

            for i, future in enumerate(asyncio.as_completed(futures), 1):
                if self.progress_callback:
                    self.progress_callback(i, len(file_paths), "Processing documents...")
                try:
                    docs = await future
                    documents.extend(docs)
                except Exception as e:
                    logger.error(f"Error in document processing task: {e}")

        return documents

    async def _create_qdrant_store(self, chunks: List[Any]) -> Qdrant:
        """Create Qdrant vector store"""
        try:
            # Create collection if it doesn't exist
            try:
                collection_info = self.client.get_collection(Config.Database.DOCUMENTS_COLLECTION)
            except UnexpectedResponse:
                self.client.recreate_collection(
                    collection_name=Config.Database.DOCUMENTS_COLLECTION,
                    **Config.Database.COLLECTION_CONFIG
                )

            vector_store = Qdrant(
                client=self.client,
                collection_name=Config.Database.DOCUMENTS_COLLECTION,
                embeddings=self.embeddings
            )

            # Add documents in batches
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                if self.progress_callback:
                    progress = min((i + batch_size) / len(chunks), 1.0)
                    self.progress_callback(
                        int(progress * 100),
                        100,
                        f"Adding documents to vector store: {i}/{len(chunks)}"
                    )
                vector_store.add_documents(batch)

            return vector_store

        except Exception as e:
            logger.error(f"Error creating Qdrant store: {e}")
            raise

    def _create_faiss_store(self, chunks: List[Any]) -> FAISS:
        """Create FAISS vector store"""
        if self.progress_callback:
            self.progress_callback(0, 1, "Creating local vector store...")

        vector_store = FAISS.from_documents(chunks, self.embeddings)

        # Save to disk if configured
        if Config.Database.SAVE_LOCAL_DB:
            vector_store.save_local(Config.Path.DATABASE_DIR / "faiss_index")

        return vector_store