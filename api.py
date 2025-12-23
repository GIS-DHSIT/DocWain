from contextlib import suppress
from pathlib import Path
from typing import AsyncGenerator, List
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
import uvicorn

from src.app.config import Config
from src.app.ingestor import Ingestor
from src.app.model import create_llm, create_reranker


def create_retriever(llm, vector_store):
    """Build a retriever with optional reranking."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    if Config.Retriever.USE_RERANKER:
        reranker = create_reranker()
        return ContextualCompressionRetriever(
            base_retriever=retriever,
            base_compressor=reranker,
        )
    return retriever


def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use the provided context to answer the question. "
                "If the answer is not contained in the context, respond with \"I don't know.\" "
                "\n\nContext:\n{context}",
            ),
            ("human", "{input}"),
        ]
    )


def create_chain(llm, retriever):
    prompt = _build_prompt()
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)


async def ask_question(chain, question: str, session_id: str) -> AsyncGenerator[str | List[Document], None]:
    """Run a retrieval chain and yield the answer followed by sources."""
    result = await chain.ainvoke({"input": question})
    answer = (
        result.get("answer")
        or result.get("output_text")
        or result.get("result")
        or str(result)
    )
    yield answer

    sources = result.get("context") or []
    if sources:
        yield sources

app = FastAPI(title="DocWain API")

# Store chains for different sessions
session_chains = {}


class Question(BaseModel):
    text: str
    session_id: str


@app.post("/api/v1/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload PDF documents and create a QA chain for the session.
    Returns a session ID for subsequent questions.
    """
    try:
        # Save files temporarily
        file_paths: List[Path] = []
        for file in files:
            file_location = Config.Path.DOCUMENTS_DIR / f"api_{uuid4().hex}_{file.filename}"
            with file_location.open("wb") as buffer:
                buffer.write(await file.read())
            file_paths.append(file_location)

        # Process files and create chain
        vector_store = Ingestor().ingest(file_paths)
        llm = create_llm()
        retriever = create_retriever(llm, vector_store=vector_store)
        chain = create_chain(llm, retriever)

        # Generate session ID and store chain
        session_id = f"session-{len(session_chains) + 1}"
        session_chains[session_id] = chain

        return {"session_id": session_id, "message": "Documents processed successfully"}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for file_path in file_paths:
            with suppress(FileNotFoundError):
                file_path.unlink()


@app.post("/api/v1/ask")
async def ask(question: Question):
    """
    Ask a question about the uploaded documents.
    Requires a session ID from a previous upload.
    """
    if question.session_id not in session_chains:
        raise HTTPException(status_code=404, detail="Session not found")

    async def generate_response():
        try:
            async for event in ask_question(
                    session_chains[question.session_id],
                    question.text,
                    question.session_id
            ):
                if isinstance(event, str):
                    yield event
                elif isinstance(event, list):
                    # For document references, yield them as JSON
                    yield "\n\nSources:\n" + "\n".join(
                        f"- {doc.page_content[:200]}..."
                        for doc in event
                    )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream"
    )


@app.delete("/api/v1/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and clean up associated resources.
    """
    if session_id not in session_chains:
        raise HTTPException(status_code=404, detail="Session not found")

    del session_chains[session_id]
    return {"message": "Session deleted successfully"}


@app.get("/api/v1/health")
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "healthy"}


# Save this file as api.py in the DocWain directory
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
