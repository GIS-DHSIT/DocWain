import openai
from backend.services.vector_store import retrieve_similar_documents

openai.api_key = "YOUR_OPENAI_API_KEY"

def generate_answer(query, response_type="detailed"):
    """
    Processes the query using GPT-4, retrieving relevant document content.
    """
    documents = retrieve_similar_documents(query)

    # Construct input prompt for GPT
    document_context = "\n\n".join([doc["content"][:2000] for doc in documents])  # Limit text
    prompt = f"""
    You are an AI assistant that answers questions based on the provided documents.

    Context from documents:
    {document_context}

    Question: {query}
    Answer:
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful assistant that provides accurate answers from documents."},
                  {"role": "user", "content": prompt}],
        temperature=0.7 if response_type == "detailed" else 0.3
    )

    return {
        "response": response["choices"][0]["message"]["content"],
        "source_documents": [doc["filename"] for doc in documents]
    }
