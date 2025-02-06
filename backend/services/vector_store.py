import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import duckdb

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize FAISS Index
index = faiss.IndexFlatL2(384)  # 384 dimensions for MiniLM

def store_document_embeddings():
    """Retrieve documents from DuckDB, create embeddings, and store them in FAISS."""
    conn = duckdb.connect("documents.duckdb")
    docs = conn.execute("SELECT id, content FROM documents").fetchall()

    embeddings = []
    doc_ids = []

    for doc_id, text in docs:
        vector = embedding_model.encode(text)
        embeddings.append(vector)
        doc_ids.append(doc_id)

    # Convert to numpy array & add to FAISS index
    embeddings_np = np.array(embeddings, dtype=np.float32)
    index.add(embeddings_np)

    return "Embeddings stored successfully"

def retrieve_similar_documents(query, top_k=3):
    """Retrieve top-k most relevant documents for a given query."""
    query_vector = embedding_model.encode(query).reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)

    conn = duckdb.connect("documents.duckdb")
    results = []
    for idx in indices[0]:
        if idx >= 0:
            doc = conn.execute(f"SELECT filename, content FROM documents WHERE id = '{doc_ids[idx]}'").fetchone()
            results.append({"filename": doc[0], "content": doc[1]})

    return results
