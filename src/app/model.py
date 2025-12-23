import os
import sys
from typing import Optional

from langchain_community.chat_models import ChatOllama
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_groq import ChatGroq

path = os.getcwd()
sys.path.append(path)

from .config import Config


def _get_groq_api_key() -> str:
    api_key: Optional[str] = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable is required when using the hosted Groq model."
        )
    return api_key


def create_llm() -> BaseLanguageModel:
    if Config.Model.USE_LOCAL:
        return ChatOllama(
            model=Config.Model.LOCAL_LLM,
            temperature=Config.Model.TEMPERATURE,
            keep_alive="1h",
            max_tokens=Config.Model.MAX_TOKENS,
        )
    else:
        return ChatGroq(
            temperature=Config.Model.TEMPERATURE,
            model_name=Config.Model.REMOTE_LLM,
            max_tokens=Config.Model.MAX_TOKENS,
            groq_api_key=_get_groq_api_key(),
        )


def create_embeddings() -> FastEmbedEmbeddings:
    threads = os.getenv("FASTEMBED_THREADS")
    return FastEmbedEmbeddings(
        model_name=Config.Model.EMBEDDINGS,
        threads=int(threads) if threads else None,
    )


def create_reranker() -> FlashrankRerank:
    return FlashrankRerank(model=Config.Model.RERANKER)
