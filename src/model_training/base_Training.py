
from typing import List
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.api.config import Config
from src.api.genai_client import generate_text, get_genai_client


# GeminiManager: handles Gemini API calls
class GeminiManager:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        if not getattr(Config.Model, "GEMINI_API_KEY", None):
            raise ValueError("GEMINI_API_KEY is not configured")
        get_genai_client(Config.Model.GEMINI_API_KEY)
        self.model_name = model_name
        self.generation_config = {
            "temperature": getattr(Config.LLM, "TEMPERATURE", 0.3),
            "top_p": getattr(Config.LLM, "TOP_P", 0.95),
        }

    def run_model(self, prompt: str) -> str:
        """Run Gemini model with the given prompt"""
        text, _ = generate_text(
            api_key=Config.Model.GEMINI_API_KEY,
            model=self.model_name,
            prompt=prompt,
            generation_config=self.generation_config,
        )
        return text


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
        from src.utils.payload_utils import get_canonical_text

        return [get_canonical_text(hit.payload or {}) for hit in hits if get_canonical_text(hit.payload or {})]


# RAGPipeline: integrates retriever + Gemini
class RAGPipeline:
    def __init__(self, retriever: QdrantRetriever, model_name: str = "gemini-2.5-flash"):
        self.retriever = retriever
        self.model_runner = GeminiManager(model_name=model_name)

    def ask(self, query: str, context_docs: int = 5) -> str:
        # Retrieve relevant documents from vector DB
        docs = self.retriever.retrieve_documents(query, top_k=context_docs)
        context = "\n\n".join(docs) if docs else "No relevant documents found."
        prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}"
        return self.model_runner.run_model(prompt)


# Setup example (won't run real inference here due to lack of actual Qdrant/Gemini access)
if __name__ == "__main__":
    rag = RAGPipeline(retriever=QdrantRetriever("67b4e5402191ec097997c2b4"))
    answer = rag.ask("summarize the content")
    print(answer)
