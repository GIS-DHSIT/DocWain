from typing import Dict, Any
from langchain.schema.runnable import RunnablePassthrough, Runnable
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseLanguageModel
from langchain.schema.retriever import BaseRetriever
from apps.core.utils.logger import setup_logger

logger = setup_logger()

def create_chain(llm: BaseLanguageModel, retriever: BaseRetriever) -> Runnable:
    """Create the QA chain"""
    if not llm or not retriever:
        raise ValueError("Both LLM and retriever must be provided")

    try:
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Answer the question using the provided context. 
            If the answer cannot be found in the context, say so.
            Use markdown formatting where appropriate.

            Context: {context}
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        # Create the chain
        retriever_chain = (
            RunnablePassthrough.assign(
                context=lambda x: retriever.get_relevant_documents(x["question"]),
                chat_history=lambda x: []
            )
        )

        chain = (
            retriever_chain
            | prompt
            | llm
        )

        return chain
    except Exception as e:
        logger.error(f"Error creating chain: {e}")
        raise
