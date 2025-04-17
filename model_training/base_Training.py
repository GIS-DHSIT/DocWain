# Step 1: Base setup for RAG-style pipeline
# We'll structure the RAG module in the following components:
# - OllamaManager: to list and use locally available models
# - QdrantRetriever: to retrieve relevant documents
# - RAGPipeline: combines query, retrieval, and generation

import subprocess
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, SearchRequest
from api.config import Config
from sentence_transformers import SentenceTransformer
import ollama


# OllamaManager: handles local model listing and querying
class OllamaManager:
    def list_models(self) -> List[str]:
        """List available Ollama models"""
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        lines = result.stdout.splitlines()[1:]  # skip header
        models = [line.split()[0] for line in lines if line.strip()]
        return models

    def run_model(self, model_name: str, prompt: str) -> str:
        """Run a model with the given prompt"""
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        return response['message']['content']


# QdrantRetriever: connects to Qdrant and fetches top-k results
class QdrantRetriever:
    def __init__(self, collection_name: str):
        self.collection = collection_name
        self.client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=60)
        self.embedding_model = SentenceTransformer(Config.Model.SENTENCE_TRANSFORMERS)

    def retrieve_documents(self, query: str, top_k: int = 5) -> List[str]:
        """Search the Qdrant collection and return top-k documents"""
        query_vector = self.embedding_model.encode(query).tolist()
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k
        )
        return [hit.payload['text'] for hit in hits if 'text' in hit.payload]


# RAGPipeline: integrates everything together
class RAGPipeline:
    def __init__(self, retriever: QdrantRetriever, model_name: str):
        self.retriever = retriever
        self.ollama = OllamaManager()
        self.model_name = model_name

    def ask(self, query: str, context_docs: int = 5) -> str:
        # Retrieve relevant documents from vector DB
        docs = self.retriever.retrieve_documents(query, top_k=context_docs)
        context = "\n\n".join(docs)
        prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}"
        return self.ollama.run_model(self.model_name, prompt)

# Setup example (won't run real inference here due to lack of actual Qdrant/Ollama access)
rag = RAGPipeline(retriever=QdrantRetriever("67b4e5402191ec097997c2b4"), model_name="mistral")
answer = rag.ask("summarize the content")
print(answer)
