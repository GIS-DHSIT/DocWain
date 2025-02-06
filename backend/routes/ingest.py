from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from backend.services.storage_handler import fetch_files_from_source
from backend.services.text_extractor import extract_text_from_document
from database.duckdb_handler import store_document_metadata

router = APIRouter()


@router.post("/ingest/")
async def ingest_document(
        source_type: str = Form(...),
        file: UploadFile = File(None),
        s3_bucket: str = Form(None),
        azure_container: str = Form(None),
        ftp_path: str = Form(None)
):
    """
    Ingest documents from multiple sources.
    """
    try:
        if file:
            # Process uploaded file
            content = await file.read()
            extracted_text = extract_text_from_document(file.filename, content)
            store_document_metadata(file.filename, extracted_text, "uploaded_file")

        elif source_type in ["s3", "azure", "ftp"]:
            # Fetch files from specified cloud storage
            files = fetch_files_from_source(source_type, s3_bucket, azure_container, ftp_path)
            for filename, file_content in files:
                extracted_text = extract_text_from_document(filename, file_content)
                store_document_metadata(filename, extracted_text, source_type)

        return {"message": "Documents ingested successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
