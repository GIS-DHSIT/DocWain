import streamlit as st
import pandas as pd
import json
import fitz
from io import BytesIO



# Function to download a file from FTP
def read_ftp_file(ftp, file_name):
    file_data = BytesIO()
    ftp.retrbinary(f"RETR {file_name}", file_data.write)
    file_data.seek(0)
    return file_data.read()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_bytes):
    text = ""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text

def ftpConn(selected_files,ftp):
    if selected_files:
        st.subheader("Extracted Content")
        for file in selected_files:
            st.write(f"### File: {file}")
            try:
                content = read_ftp_file(file, ftp)

                # Auto detect content type and display accordingly
                if file.endswith(".csv"):
                    df = pd.read_csv(BytesIO(content))
                    st.dataframe(df)
                elif file.endswith(".json"):
                    parsed_json = json.loads(content.decode("utf-8"))
                    st.json(parsed_json)
                elif file.endswith(".pdf"):
                    extracted_text = extract_text_from_pdf(BytesIO(content))
                    st.text_area("Extracted PDF Content:", extracted_text, height=300)
                else:
                    st.text_area("Content:", content.decode("utf-8"), height=300)

            except Exception as e:
                st.error(f"Error reading file {file}: {e}")
