from fastapi import APIRouter, UploadFile, File, HTTPException, Form
import os
import shutil
from database.duckdb_handler import list_documents, tag_document, get_document_path

router = APIRouter()

UPLOAD_FOLDER = "uploaded_docs"

# Ensure upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# ✅ Upload Document API
@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"message": f"File '{file.filename}' uploaded successfully!"}


# ✅ List Documents API
@router.get("/list")
def list_docs():
    return list_documents()


# ✅ Tag Document API
@router.post("/tag")
def tag_doc(doc_id: str = Form(...), tag: str = Form(...)):
    if tag_document(doc_id, tag):
        return {"message": f"Document {doc_id} tagged as {tag}"}
    raise HTTPException(status_code=400, detail="Failed to tag document")


# ✅ Download Document API
@router.get("/download/{doc_id}")
def download_document(doc_id: str):
    file_path = get_document_path(doc_id)

    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return {"download_url": f"/static/{os.path.basename(file_path)}"}
