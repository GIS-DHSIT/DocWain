from pathlib import Path
from typing import Iterable, List
import os,sys

from langchain_community.document_loaders import PyPDFium2Loader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter

path = os.getcwd()
sys.path.append(path)

from .config import Config
from .model import create_embeddings


class Ingestor:
    def __init__(self):
        self.embeddings = create_embeddings()
        self.semantic_splitter = SemanticChunker(
            self.embeddings, breakpoint_threshold_type="interquartile"
        )
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=128,
            add_start_index=True,
        )

    def _load_documents(self, doc_path: Path) -> List[Document]:
        loaded_documents = PyPDFium2Loader(str(doc_path)).load()
        documents: List[Document] = []
        for doc in loaded_documents:
            text = doc.page_content.strip()
            if not text:
                continue
            metadata = dict(doc.metadata or {})
            metadata.setdefault("source", str(doc_path))
            documents.append(Document(page_content=text, metadata=metadata))
        return documents

    def _semantic_expand(self, documents: Iterable[Document]) -> List[Document]:
        combined_text = "\n".join(doc.page_content for doc in documents).strip()
        if not combined_text:
            return []
        try:
            return self.semantic_splitter.create_documents([combined_text])
        except ValueError:
            # Semantic splitter can fail on very small inputs; fall back to basic chunks.
            return [Document(page_content=combined_text)]

    def ingest(self, doc_paths: List[Path]) -> VectorStore:
        documents: List[Document] = []
        for doc_path in doc_paths:
            loaded_documents = self._load_documents(doc_path)
            semantic_documents = self._semantic_expand(loaded_documents)
            if not semantic_documents:
                continue
            base_documents = [
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "source": str(doc_path)},
                )
                for doc in semantic_documents
            ]
            documents.extend(
                self.recursive_splitter.split_documents(base_documents)
            )

        if not documents:
            raise ValueError(
                "No textual content found in the provided documents. Ensure the files contain readable text."
            )

        return Qdrant.from_documents(
            documents=documents,
            embedding=self.embeddings,
            path=Config.Path.DATABASE_DIR,
            collection_name=Config.Database.DOCUMENTS_COLLECTION,
        )
