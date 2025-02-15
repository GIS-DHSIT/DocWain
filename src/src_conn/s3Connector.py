import streamlit as st
import boto3
import pandas as pd
import json
import fitz  # PyMuPDF for PDF extraction


def extracts3(s3):
    # List S3 Buckets
    buckets = [bucket["Name"] for bucket in s3.list_buckets()["Buckets"]]
    selected_bucket = st.selectbox("Select an S3 Bucket:", buckets)

    if selected_bucket:
        # List Files in the Selected Bucket
        objects = s3.list_objects_v2(Bucket=selected_bucket)

        if "Contents" in objects:
            file_list = [obj["Key"] for obj in objects["Contents"]]
            selected_files = st.multiselect("Select Files to Extract Content:", file_list)

            # Function to read file content
            def read_s3_file(bucket, file_key):
                obj = s3.get_object(Bucket=bucket, Key=file_key)
                content = obj["Body"].read()  # Read as binary
                return content

            # Function to extract text from a PDF file
            def extract_text_from_pdf(pdf_bytes):
                text = ""
                with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                    for page in doc:
                        text += page.get_text("text") + "\n"
                return text

            if selected_files:
                st.subheader("Extracted Content")
                for file in selected_files:
                    st.write(f"### File: {file}")
                    try:
                        content = read_s3_file(selected_bucket, file)

                        # Auto detect content type and display accordingly
                        if file.endswith(".csv"):
                            df = pd.read_csv(content)
                            st.dataframe(df)
                        elif file.endswith(".json"):
                            parsed_json = json.loads(content.decode("utf-8"))
                            st.json(parsed_json)
                        elif file.endswith(".pdf"):
                            extracted_text = extract_text_from_pdf(content)
                            st.text_area("Extracted PDF Content:", extracted_text, height=300)
                        else:
                            st.text_area("Content:", content.decode("utf-8"), height=300)

                    except Exception as e:
                        st.error(f"Error reading file {file}: {e}")
        else:
            st.warning("No files found in the selected bucket.")
