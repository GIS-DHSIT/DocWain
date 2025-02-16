import ollama
import faiss
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from bson.objectid import ObjectId
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from api.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL = SentenceTransformer(Config.Model.SENTENCE_TRANSFORMERS)
mongoClient = MongoClient(Config.MongoDB.URI)


app = FastAPI(title="DocWain API")

class QuestionRequest(BaseModel):
    query: str
    mongo_db_name: str = 'test'
    mongo_collection:str = 'documents'
    profile_id: str = "default"
    model_name: str = "llama3.2"


def load_embeddings_from_mongo(db,tags):
    """Loads FAISS index and text data from MongoDB."""
    logging.info(f"Loading embeddings from MongoDB for model: {tags}")
    embedding_data = db.find_one({"profile": ObjectId(tags)})
    if not embedding_data:
        logging.warning("No embeddings found in database.")
        return None

    embeddings = np.array(embedding_data["embeddings"], dtype=np.float32)
    texts = embedding_data["texts"]

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


@app.post("/ask")
def ask_question_api(request: QuestionRequest):
    """API endpoint for answering questions."""
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")
    answer = answer_question(request.query, request.mongo_db_name, request.mongo_collection,
                             request.profile_id, request.model_name)
    return {"answer": answer}

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
