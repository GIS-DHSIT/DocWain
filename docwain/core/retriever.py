from typing import Optional
from langchain_core.language_models import BaseLanguageModel
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.vectorstores import VectorStore
import os,sys
path = os.getcwd()
sys.path.append(path)
from docwain.config import Config
from docwain.utils.logger import setup_logger

logger = setup_logger()


def create_retriever(
        llm: BaseLanguageModel,
        vector_store: VectorStore,
        use_compression: bool = Config.Retriever.USE_CHAIN_FILTER
) -> ContextualCompressionRetriever:
    """Create document retriever with optional compression"""
    try:
        # Create base retriever from vector store
        base_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": Config.Retriever.TOP_K,
                "score_threshold": Config.Retriever.SIMILARITY_THRESHOLD
            }
        )

        if use_compression:
            # Create compressor for more focused retrieval
            compressor = LLMChainExtractor.from_llm(llm)

            # Return compressed retriever
            return ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )

        return base_retriever

    except Exception as e:
        logger.error(f"Error creating retriever: {e}")
        raise