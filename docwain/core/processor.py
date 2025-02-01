from typing import Optional, List, Dict, Any
import asyncio
from pathlib import Path
from langchain_core.language_models import BaseLanguageModel
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
import os,sys
path = os.getcwd()
sys.path.append(path)
from docwain.config import Config
from docwain.utils.logger import setup_logger

logger = setup_logger()

def create_llm() -> BaseLanguageModel:
    """Create language model instance based on configuration"""
    try:
        if Config.Model.USE_LOCAL:
            return ChatOllama(
                model=Config.Model.LOCAL_LLM,
                temperature=Config.Model.TEMPERATURE,
                max_tokens=Config.Model.MAX_TOKENS,
            )
        else:
            return ChatGroq(
                temperature=Config.Model.TEMPERATURE,
                model_name=Config.Model.REMOTE_LLM,
                max_tokens=Config.Model.MAX_TOKENS,
            )
    except Exception as e:
        logger.error(f"Error creating LLM: {e}")
        raise