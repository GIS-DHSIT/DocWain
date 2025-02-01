from typing import Dict, Any
from langchain.schema.runnable import RunnablePassthrough, Runnable
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseLanguageModel
from langchain.schema.retriever import BaseRetriever
import os,sys
path = os.getcwd()
sys.path.append(path)
from docwain.utils.logger import setup_logger

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

        def extract_question(input_data: Dict[str, Any]) -> str:
            """Extract question from input data"""
            if isinstance(input_data, dict):
                return str(input_data.get("question", ""))
            return str(input_data)

        # Create the chain
        retriever_chain = (
                RunnablePassthrough.assign(
                    question=lambda x: extract_question(x)
                )
                | {
                    "context": lambda x: retriever.get_relevant_documents(x["question"]),
                    "question": lambda x: x["question"],
                    "chat_history": lambda x: []
                }
        )

        chain = (
                retriever_chain
                | prompt
                | llm
                | StrOutputParser()
        )

        return chain
    except Exception as e:
        logger.error(f"Error creating chain: {e}")
        raise