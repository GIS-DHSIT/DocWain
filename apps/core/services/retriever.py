from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.language_models import BaseLanguageModel
from langchain.vectorstores import VectorStore
from apps.core.utils.logger import setup_logger

logger = setup_logger()

def create_retriever(
    llm: BaseLanguageModel,
    vector_store: VectorStore,
    use_compression: bool = False
) -> ContextualCompressionRetriever:
    """Create document retriever"""
    try:
        # Create base retriever
        base_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
                "score_threshold": 0.7
            }
        )

        if use_compression:
            compressor = LLMChainExtractor.from_llm(llm)
            return ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )

        return base_retriever

    except Exception as e:
        logger.error(f"Error creating retriever: {e}")
        raise