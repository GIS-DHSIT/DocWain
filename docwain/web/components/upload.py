import streamlit as st
from typing import List, Optional
from streamlit.runtime.uploaded_file_manager import UploadedFile


def show_upload_section() -> Optional[List[UploadedFile]]:
    """Display the document upload section"""
    st.header("Upload Documents")

    upload_type = st.radio(
        "Choose upload method:",
        ["File Upload", "Remote Source"]
    )

    if upload_type == "File Upload":
        return st.file_uploader(
            "Upload your documents",
            type=["pdf", "docx", "doc", "xlsx", "xls", "csv"],
            accept_multiple_files=True,
            help="Supported formats: PDF, Word, Excel, CSV"
        )
    else:
        with st.form("remote_source"):
            source_type = st.selectbox(
                "Source type",
                ["S3", "FTP"]
            )

            if source_type == "S3":
                bucket = st.text_input("Bucket name")
                key = st.text_input("File key (path)")

                if st.form_submit_button("Connect"):
                    if bucket and key:
                        return [{"type": "s3", "bucket": bucket, "key": key}]

            else:  # FTP
                host = st.text_input("FTP host")
                path = st.text_input("File path")
                username = st.text_input("Username (optional)")
                password = st.text_input("Password (optional)", type="password")

                if st.form_submit_button("Connect"):
                    if host and path:
                        return [{
                            "type": "ftp",
                            "host": host,
                            "path": path,
                            "username": username,
                            "password": password
                        }]

    return None