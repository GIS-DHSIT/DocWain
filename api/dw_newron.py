import ollama
import faiss
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from bson.objectid import ObjectId
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AzureOpenAI
import uvicorn
from api.config import Config
from api.dataHandler import trainData

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL = SentenceTransformer(Config.Model.SENTENCE_TRANSFORMERS)
AzureOpenAI.api_key = Config.Model.AZURE_OPENAI_API_KEY
mongoClient = MongoClient(Config.MongoDB.URI)
chunks_collection = mongoClient[Config.MongoDB.DB]["embeddings_chunks"]

app = FastAPI(title="DocWain API")

class QuestionRequest(BaseModel):
    query: str
    mongo_db_name: str = 'test'
    mongo_collection:str = 'documents'
    profile_id: str = "default"
    model_name: str = "llama3.2"

class AzureQuestionRequest(BaseModel):
    query: str
    mongo_db_name: str = 'test'
    mongo_collection:str = 'documents'
    profile_id: str = "default"
    model_name: str = "OpenAI"
    api_version: str = "2023-07-01-preview"


class TrainRequest(BaseModel):
    collectionName: str = 'documents'
    collectionDir: str = 'connectors'
    schemaName: str = 'actual documents provided '


def load_embeddings_from_mongo(db, tags):
    """Loads FAISS index and text data from MongoDB, reconstructing chunked embeddings."""
    logging.info(f"Loading embeddings from MongoDB for model: {tags}")
    embedding_data = db.find_one({"profile": ObjectId(tags)})

    if not embedding_data or "embedding_chunks" not in embedding_data:
        logging.warning("No embeddings found in database.")
        return None

    # Load and merge chunked embeddings
    chunk_ids = embedding_data["embedding_chunks"]
    embeddings = []

    for chunk_id in chunk_ids:
        chunk_doc = chunks_collection.find_one({"_id": chunk_id})
        if chunk_doc:
            embeddings.extend(chunk_doc["chunk"])

    embeddings = np.array(embeddings, dtype=np.float32)
    texts = embedding_data["texts"]

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return {"index": index, "texts": texts}




def answer_question(query, mongoDbName, collection_name,tag="default",model='llama3.2'):
    """Answers a question based on stored embeddings."""
    db = mongoClient[mongoDbName]
    conColl = db[collection_name]
    embeddings = load_embeddings_from_mongo(conColl,tag)
    if embeddings is None:
        return "No trained model found!"
    logging.info(f"Processing query: {query} for model: {tag}")
    query_embedding = MODEL.encode([query], convert_to_numpy=True)
    D, I = embeddings["index"].search(query_embedding, 3)
    retrieved_text = "\n".join([embeddings["texts"][i] for i in I[0]])
    logging.info(f"Retrieving response using Ollama model {model}.")
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Answer based on this context: {retrieved_text}\nQuestion: {query}"}
        ]
    )
    return response["message"]


def azure_answer_question(query, mongoDbName, collection_name,tag="default",model="gpt-35-turbo",apiVersion ="2023-07-01-preview"):
    """Answers a question based on stored embeddings."""
    db = mongoClient[mongoDbName]
    conColl = db[collection_name]
    embeddings = load_embeddings_from_mongo(conColl,tag)
    if embeddings is None:
        return "No trained model found!"
    logging.info(f"Processing query: {query} for model: {tag}")
    query_embedding = MODEL.encode([query], convert_to_numpy=True)
    D, I = embeddings["index"].search(query_embedding, 3)
    retrieved_text = "\n".join([embeddings["texts"][i] for i in I[0]])
    logging.info(f"Retrieving response using Azure OpenAI {model}.")
    client = AzureOpenAI(
        api_version=apiVersion,
        api_key=Config.Model.AZURE_OPENAI_API_KEY,
        azure_endpoint=Config.Model.AZURE_OPENAI_ENDPOINT,
    )
    response = client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Answer based on this context: {retrieved_text}\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message.content

@app.post("/ask")
def ask_question_api(request: QuestionRequest):
    """API endpoint for answering questions."""
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")
    answer = answer_question(request.query, request.mongo_db_name, request.mongo_collection,
                             request.profile_id, request.model_name)
    return {"answer": answer}


@app.post("/askAzure")
def ask_question_api(request: AzureQuestionRequest):
    """Azure openAI API endpoint for answering questions."""
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")
    answer = azure_answer_question(request.query, request.mongo_db_name, request.mongo_collection,
                             request.profile_id, request.model_name,request.api_version)
    return {"answer": answer}


@app.post("/train")
def trigger_training(request: TrainRequest):
    """
    API endpoint to trigger document training.

    Request Body:
    - collectionName: Name of the MongoDB collection
    - bucketName: Name of the bucket where documents are stored

    Response:
    - Training status message
    """
    try:
        logging.info(
            f"Received training request for collection: {request.collectionName}, connector: {request.collectionDir}")

        status_response = trainData(request.collectionDir, request.schemaName,request.collectionName)

        logging.info("Training completed successfully.")
        return {"status": "success", "message": "Training process triggered", "response": "Completed"}

    except Exception as e:
        logging.error(f"Training API error: {e}")
        raise HTTPException(status_code=500, detail="Training process failed")


@app.get("/models")
def list_available_models():
    """API endpoint to list locally available models."""
    try:
        models = ollama.list().model_dump()
        return {"models": models}
    except Exception as e:
        logging.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve available models")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
