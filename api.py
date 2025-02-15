from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import asyncio
import uvicorn
import os
import sys
path = os.getcwd()
sys.path.append(path)
from src.chain import ask_question, create_chain
from src.config import Config
from src.ingestor import Ingestor
from src.model import create_llm
from src.retriever import create_retriever
from src.uploader import upload_files

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
        file_paths = []
        for file in files:
            file_location = f"temp_{file.filename}"
            with open(file_location, "wb") as buffer:
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


def run():
    return uvicorn.run(app, host="0.0.0.0", port=80)