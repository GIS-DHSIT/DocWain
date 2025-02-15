from typing import List, Optional
from qdrant_client import QdrantClient
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import PointStruct, VectorParams, Distance
from src.web.webConfig import Config
import uuid

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
        self.qdrant_client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API)
        self.collection_name = Config.Database.DOCUMENTS_COLLECTION

        # Ensure the collection exists with the correct vector configuration
        self._initialize_qdrant_collection()

    def _initialize_qdrant_collection(self):
        """
        Ensures the Qdrant collection is created with the correct configuration.
        If it exists, it checks for compatibility.
        """
        try:
            # Check if collection exists
            existing_collection = self.qdrant_client.get_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' already exists.")

        except Exception:
            print(f"Creating collection: {self.collection_name}")

            # Dynamically determine embedding vector size
            vector_size = len(self.embeddings.embed_documents("sample text"))

            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,  # Dynamically detected vector size
                    distance=Distance.COSINE
                )
            )

    def ingest_extracted(self, extracted_contents: List[str], group: Optional[str] = "default") -> None:
        """
        Ingest extracted text content with an optional group classification.
        """
        documents = []
        for content in extracted_contents:
            documents.extend(
                self.recursive_splitter.split_documents(
                    self.semantic_splitter.create_documents([content])
                )
            )

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=self.embeddings.embed_documents(doc.page_content),  # Convert text to embedding
                payload={"text": doc.page_content, "group": group}  # Store group metadata
            )
            for doc in documents
        ]

        self.qdrant_client.upsert(collection_name=self.collection_name, points=points)
        print(f"Ingested {len(points)} documents into group '{group}'.")

    def search_by_group(self, query: str, group: str, top_k: int = 5):
        """
        Perform a vector search within a specific group.
        """
        query_vector = self.embeddings.embed(query)

        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter={"must": [{"key": "group", "match": {"value": group}}]}
        )

        return [{"text": res.payload["text"], "score": res.score} for res in results]

