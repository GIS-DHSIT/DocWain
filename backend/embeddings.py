import faiss
import numpy as np
from llama_index import SimpleDirectoryReader, ServiceContext, GPTVectorStoreIndex
from langchain.embeddings.openai import OpenAIEmbeddings
import os

INDEX_PATH = "models/index.faiss"
DIMENSIONS = 768  # Adjust based on embedding model
embeddings = OpenAIEmbeddings()

# Load index or create new one
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
else:
    index = faiss.IndexFlatL2(DIMENSIONS)

def process_document(filepath):
    docs = SimpleDirectoryReader(filepath).load_data()
    vectors = [embeddings.embed(doc.text) for doc in docs]
    index.add(np.array(vectors))
    faiss.write_index(index, INDEX_PATH)

def query_ai(question):
    query_vector = np.array([embeddings.embed(question)])
    _, I = index.search(query_vector, 1)
    return f"Best match found in document {I[0][0]}"
