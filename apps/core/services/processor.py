from langchain.chat_models import ChatOllama
from langchain_core.language_models import BaseLanguageModel
from django.conf import settings
from apps.core.utils.logger import setup_logger
logger = setup_logger()

def create_llm() -> BaseLanguageModel:
    """Create language model instance"""
    try:
        return ChatOllama(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE
        )
    except Exception as e:
        logger.error(f"Error creating LLM: {e}")
        raise