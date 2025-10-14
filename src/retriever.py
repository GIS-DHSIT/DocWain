from typing import Optional

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.chain_filter import LLMChainFilter
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_qdrant import Qdrant

from .config import Config
from .model import create_embeddings, create_reranker


def create_retriever(
    llm: BaseLanguageModel, vector_store: Optional[VectorStore] = None
) -> VectorStoreRetriever:
    if not vector_store:
        if Config.Database.QDRANT_ENABLED:
            vector_store = Qdrant.from_existing_collection(
                embedding=create_embeddings(),
                collection_name=Config.Database.DOCUMENTS_COLLECTION,
                host=Config.Database.QDRANT_HOST,
                port=Config.Database.QDRANT_PORT,
                api_key=Config.Database.QDRANT_API_KEY,
            )
        else:
            vector_store = Qdrant.from_existing_collection(
                embedding=create_embeddings(),
                collection_name=Config.Database.DOCUMENTS_COLLECTION,
                path=Config.Path.DATABASE_DIR,
            )

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )

    if Config.Retriever.USE_RERANKER:
        retriever = ContextualCompressionRetriever(
            base_compressor=create_reranker(), base_retriever=retriever
        )

    if Config.Retriever.USE_CHAIN_FILTER:
        retriever = ContextualCompressionRetriever(
            base_compressor=LLMChainFilter.from_llm(llm), base_retriever=retriever
        )

    return retriever
