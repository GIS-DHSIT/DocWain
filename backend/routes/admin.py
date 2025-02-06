from fastapi import APIRouter, HTTPException
from database.duckdb_handler import get_pending_users, approve_user, get_documents, tag_document

router = APIRouter()

# Fetch Pending User Approvals
@router.get("/users")
def get_users():
    users = get_pending_users()
    return users

# Approve User
@router.post("/approve/{user_id}")
def approve(user_id: str):
    if approve_user(user_id):
        return {"message": "User approved successfully"}
    raise HTTPException(status_code=404, detail="User not found")

# Fetch Uploaded Documents
@router.get("/documents")
def fetch_documents():
    docs = get_documents()
    return docs

# Tag Document
@router.post("/tag_document")
def tag_doc(doc_id: str, tag: str):
    if tag_document(doc_id, tag):
        return {"message": "Document tagged successfully"}
    raise HTTPException(status_code=404, detail="Document not found")
