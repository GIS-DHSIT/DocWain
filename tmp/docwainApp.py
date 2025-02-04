import streamlit as st
import requests
import json
import os
from datetime import datetime
import pandas as pd
import boto3
from ftplib import FTP
from dotenv import load_dotenv


class DocwainApp:
    def __init__(self):
        st.set_page_config(
            page_title="Docwain",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        self.api_url = "http://localhost:8000"
        self.initialize_session_state()

    def initialize_session_state(self):
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'role' not in st.session_state:
            st.session_state.role = None
        if 'documents' not in st.session_state:
            st.session_state.documents = []
        if 'configured_sources' not in st.session_state:
            st.session_state.configured_sources = []
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

    def login_page(self):
        st.title("Welcome to Docwain")

        col1, col2 = st.columns(2)

        with col1:
            st.header("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            role = st.selectbox("Role", ["User", "Admin"])

            if st.button("Login"):
                # In a real application, implement proper authentication
                if role == "Admin" and password == "admin123":  # Replace with secure authentication
                    st.session_state.authenticated = True
                    st.session_state.role = "admin"
                    st.experimental_rerun()
                elif role == "User" and password == "user123":  # Replace with secure authentication
                    st.session_state.authenticated = True
                    st.session_state.role = "user"
                    st.experimental_rerun()
                else:
                    st.error("Invalid credentials")

        with col2:
            st.image("https://via.placeholder.com/400x300", caption="Docwain Logo")

    def render_admin_sidebar(self):
        with st.sidebar:
            st.title("Admin Dashboard")
            st.markdown("---")
            menu_options = ["Document Management", "Source Configuration", "Document Status"]
            selected_menu = st.radio("Navigation", menu_options)

            if st.button("Logout"):
                self.logout()

            return selected_menu

    def render_user_sidebar(self):
        with st.sidebar:
            st.title("Document Chat")
            st.markdown("---")

            try:
                response = requests.get(f"{self.api_url}/documents")
                if response.status_code == 200:
                    documents = response.json()["documents"]
                    if documents:
                        st.write("Available Documents:")
                        for doc in documents:
                            st.info(f"📄 {doc['name']}")
                    else:
                        st.warning("No documents available")
            except Exception as e:
                st.error(f"Error loading documents: {str(e)}")

            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.experimental_rerun()

            if st.button("Logout"):
                self.logout()

    def admin_document_management(self):
        st.header("Document Management")

        uploaded_files = st.file_uploader(
            "Upload Documents",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'xlsx', 'csv', 'ppt', 'pptx']
        )

        if uploaded_files:
            for file in uploaded_files:
                try:
                    files = {"file": file}
                    response = requests.post(f"{self.api_url}/upload-document", files=files)

                    if response.status_code == 200:
                        document_info = {
                            "name": file.name,
                            "size": file.size,
                            "type": file.type,
                            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "status": "Active"
                        }
                        st.session_state.documents.append(document_info)
                        st.success(f"Successfully processed: {file.name}")
                    else:
                        st.error(f"Failed to process: {file.name}")
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")

    def admin_source_configuration(self):
        st.header("Configure Document Source")

        source_type = st.selectbox(
            "Select Source Type",
            ["Local Directory", "FTP Server", "AWS S3"]
        )

        if source_type == "Local Directory":
            self.configure_local_directory()
        elif source_type == "FTP Server":
            self.configure_ftp_server()
        elif source_type == "AWS S3":
            self.configure_s3_bucket()

    def configure_local_directory(self):
        path = st.text_input("Enter Directory Path")
        if st.button("Configure Local Directory"):
            if os.path.exists(path):
                self.save_source_config("local", path)
                st.success(f"Local directory configured: {path}")
            else:
                st.error("Directory does not exist")

    def configure_ftp_server(self):
        col1, col2, col3 = st.columns(3)
        with col1:
            host = st.text_input("Host")
        with col2:
            username = st.text_input("Username")
        with col3:
            password = st.text_input("Password", type="password")

        if st.button("Configure FTP"):
            connection_string = f"{host}:{username}:{password}"
            self.save_source_config("ftp", connection_string)

    def configure_s3_bucket(self):
        col1, col2, col3 = st.columns(3)
        with col1:
            bucket = st.text_input("Bucket Name")
        with col2:
            access_key = st.text_input("Access Key")
        with col3:
            secret_key = st.text_input("Secret Key", type="password")

        if st.button("Configure S3"):
            connection_string = f"{bucket}:{access_key}:{secret_key}"
            self.save_source_config("s3", connection_string)

    def admin_document_status(self):
        st.header("Document Status")

        if st.session_state.documents:
            df = pd.DataFrame(st.session_state.documents)
            st.dataframe(df)

            if st.button("Export Document List"):
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    "document_status.csv",
                    "text/csv",
                    key='download-csv'
                )
        else:
            st.info("No documents have been uploaded yet")

    def user_chat_interface(self):
        st.header("Document Chat Assistant")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask me about the documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            try:
                response = requests.post(
                    f"{self.api_url}/chat",
                    json={
                        "message": prompt,
                        "chat_history": st.session_state.chat_history
                    }
                )

                if response.status_code == 200:
                    ai_response = response.json()["response"]
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    with st.chat_message("assistant"):
                        st.markdown(ai_response)
                    st.session_state.chat_history.append((prompt, ai_response))
                else:
                    st.error("Failed to get response from the assistant")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    def save_source_config(self, source_type, connection_string):
        try:
            response = requests.post(
                f"{self.api_url}/configure-source",
                json={"source_type": source_type, "connection_string": connection_string}
            )
            if response.status_code == 200:
                config = {
                    "type": source_type,
                    "connection": connection_string,
                    "configured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.configured_sources.append(config)
                st.success(f"{source_type.title()} configured successfully")
            else:
                st.error(f"Failed to configure {source_type}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

    def logout(self):
        st.session_state.authenticated = False
        st.session_state.role = None
        st.experimental_rerun()

    def main(self):
        if not st.session_state.authenticated:
            self.login_page()
        else:
            if st.session_state.role == "admin":
                selected_menu = self.render_admin_sidebar()
                if selected_menu == "Document Management":
                    self.admin_document_management()
                elif selected_menu == "Source Configuration":
                    self.admin_source_configuration()
                else:
                    self.admin_document_status()
            else:
                self.render_user_sidebar()
                self.user_chat_interface()


if __name__ == "__main__":
    app = DocwainApp()
    app.main()