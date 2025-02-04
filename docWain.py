import asyncio
import random
import os
from pathlib import Path
import boto3
from ftplib import FTP
import tempfile
import re
import json

import streamlit as st
from dotenv import load_dotenv

from src.chain import ask_question, create_chain
from src.config import Config
from src.ingestor import Ingestor
from src.model import create_llm
from src.retriever import create_retriever
from src.uploader import upload_files

load_dotenv()

LOADING_MESSAGES = [
    "Collecting the answers from multiverse...",
    "Assembling quantum entanglement...",
    "Summoning star wisdom... almost there!",
    "Consulting Schrödinger's cat...",
    "Warping spacetime for your response...",
    "Balancing neutron star equations...",
    "Analyzing dark matter... please wait...",
    "Engaging hyperdrive... en route!",
    "Gathering photons from a galaxy...",
    "Beaming data from Andromeda... stand by!",
]


def initialize_session_state():
    if 'ftp_connections' not in st.session_state:
        st.session_state.ftp_connections = []
    if 's3_connections' not in st.session_state:
        st.session_state.s3_connections = []
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi! What do you want to know about your documents?",
            }
        ]


class DocumentSource:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()

    async def get_from_directory(self, directory_path):
        """Get PDF files from local directory"""
        pdf_files = []
        directory = Path(directory_path)
        if directory.exists() and directory.is_dir():
            for file in directory.glob("**/*.pdf"):
                pdf_files.append(str(file))
        return pdf_files

    async def get_from_ftp(self, ftp_host, ftp_user, ftp_password, remote_path):
        """Get PDF files from FTP server"""
        pdf_files = []
        try:
            ftp = FTP(timeout=30)
            ftp.connect(host=ftp_host, port=21)
            ftp.login(user=ftp_user, passwd=ftp_password)

            def process_file(name):
                try:
                    if name.lower().endswith('.pdf'):
                        local_path = os.path.join(self.temp_dir, name)
                        with open(local_path, 'wb') as local_file:
                            ftp.retrbinary(f'RETR {name}', local_file.write)
                        pdf_files.append(local_path)
                except Exception as e:
                    st.error(f"Error processing file {name}: {str(e)}")

            try:
                ftp.cwd(remote_path)
                files = []
                ftp.retrlines('NLST', files.append)

                for file in files:
                    if file.lower().endswith('.pdf'):
                        process_file(file)

            except Exception as e:
                st.error(f"Error accessing remote path: {str(e)}")
            finally:
                ftp.quit()

        except Exception as e:
            st.error(f"FTP connection error: {str(e)}")

        return pdf_files

    async def get_from_s3(self, bucket_name, prefix=""):
        """Get PDF files from S3 bucket"""
        pdf_files = []
        try:
            try:
                s3 = boto3.client('s3')
            except Exception as e:
                st.error("Failed to initialize S3 client. Please check your AWS credentials.")
                st.error(f"Error details: {str(e)}")
                return pdf_files

            try:
                s3.head_bucket(Bucket=bucket_name)
            except s3.exceptions.ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == '404':
                    st.error(f"Bucket '{bucket_name}' does not exist.")
                elif error_code == '403':
                    st.error(f"Access denied to bucket '{bucket_name}'. Please check your permissions.")
                else:
                    st.error(f"Error accessing bucket '{bucket_name}': {str(e)}")
                return pdf_files

            found_files = False
            paginator = s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if obj['Key'].lower().endswith('.pdf'):
                            found_files = True
                            try:
                                local_path = os.path.join(self.temp_dir, os.path.basename(obj['Key']))
                                s3.download_file(bucket_name, obj['Key'], local_path)
                                pdf_files.append(local_path)
                                st.success(f"Successfully downloaded: {obj['Key']}")
                            except Exception as e:
                                st.error(f"Error downloading file {obj['Key']}: {str(e)}")

            if not found_files:
                st.warning(f"No PDF files found in bucket '{bucket_name}'{f' with prefix {prefix}' if prefix else ''}")

        except Exception as e:
            st.error(f"S3 connection error: {str(e)}")
            st.error("Please ensure you have proper AWS credentials configured.")

        return pdf_files


def validate_s3_bucket_name(bucket_name):
    bucket_pattern = r"^[a-zA-Z0-9.-_]{1,255}$"
    return re.match(bucket_pattern, bucket_name) is not None


def show_ftp_manager():
    st.subheader("FTP Connections")

    # Add new FTP connection
    with st.expander("Add New FTP Connection", expanded=False):
        with st.form("ftp_form"):
            host = st.text_input("FTP Host")
            user = st.text_input("FTP Username")
            password = st.text_input("FTP Password", type="password")
            path = st.text_input("Remote Path", value="/")
            connection_name = st.text_input("Connection Name (optional)")

            if st.form_submit_button("Add FTP Connection"):
                if not all([host, user, password]):
                    st.error("Please fill all required fields!")
                else:
                    connection = {
                        'name': connection_name or host,
                        'host': host,
                        'user': user,
                        'password': password,
                        'path': path
                    }
                    st.session_state.ftp_connections.append(connection)
                    st.success("FTP connection added successfully!")

    # Show existing connections
    if st.session_state.ftp_connections:
        st.write("Existing FTP Connections:")
        for idx, conn in enumerate(st.session_state.ftp_connections):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📁 {conn['name']}")
                with col2:
                    if st.button("Remove", key=f"remove_ftp_{idx}"):
                        st.session_state.ftp_connections.pop(idx)
                        st.rerun()


def show_s3_manager():
    st.subheader("S3 Buckets")

    # Add new S3 bucket
    with st.expander("Add New S3 Bucket", expanded=False):
        with st.form("s3_form"):
            bucket_name = st.text_input("S3 Bucket Name")
            prefix = st.text_input("S3 Prefix (optional)")
            connection_name = st.text_input("Connection Name (optional)")

            if st.form_submit_button("Add S3 Bucket"):
                if not bucket_name:
                    st.error("Please enter a bucket name!")
                elif not validate_s3_bucket_name(bucket_name):
                    st.error("""
                        Invalid bucket name! Bucket names must:
                        - Be between 1 and 255 characters
                        - Contain only letters, numbers, periods, hyphens, and underscores
                        - Not be formatted as an IP address
                    """)
                else:
                    connection = {
                        'name': connection_name or bucket_name,
                        'bucket': bucket_name,
                        'prefix': prefix
                    }
                    st.session_state.s3_connections.append(connection)
                    st.success("S3 bucket added successfully!")

    # Show existing connections
    if st.session_state.s3_connections:
        st.write("Existing S3 Buckets:")
        for idx, conn in enumerate(st.session_state.s3_connections):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"🪣 {conn['name']}")
                with col2:
                    if st.button("Remove", key=f"remove_s3_{idx}"):
                        st.session_state.s3_connections.pop(idx)
                        st.rerun()


@st.cache_resource(show_spinner=False)
def build_qa_chain(files=None, directory=None):
    doc_source = DocumentSource()
    file_paths = []

    if files:
        file_paths.extend(upload_files(files))

    if directory:
        dir_files = asyncio.run(doc_source.get_from_directory(directory))
        file_paths.extend(dir_files)

    # Process FTP connections
    for ftp_config in st.session_state.ftp_connections:
        ftp_files = asyncio.run(
            doc_source.get_from_ftp(
                ftp_config['host'],
                ftp_config['user'],
                ftp_config['password'],
                ftp_config['path']
            )
        )
        file_paths.extend(ftp_files)

    # Process S3 connections
    for s3_config in st.session_state.s3_connections:
        s3_files = asyncio.run(
            doc_source.get_from_s3(
                s3_config['bucket'],
                s3_config.get('prefix', '')
            )
        )
        file_paths.extend(s3_files)

    if not file_paths:
        raise ValueError("No PDF files found from any source")

    vector_store = Ingestor().ingest(file_paths)
    llm = create_llm()
    retriever = create_retriever(llm, vector_store=vector_store)
    return create_chain(llm, retriever)


async def ask_chain(question: str, chain):
    full_response = ""
    assistant = st.chat_message(
        "assistant", avatar=str(Config.Path.IMAGES_DIR / "assistant-avatar.png")
    )
    with assistant:
        message_placeholder = st.empty()
        message_placeholder.status(random.choice(LOADING_MESSAGES), state="running")
        documents = []
        async for event in ask_question(chain, question, session_id="session-id-42"):
            if type(event) is str:
                full_response += event
                message_placeholder.markdown(full_response)
            if type(event) is list:
                documents.extend(event)
        for i, doc in enumerate(documents):
            with st.expander(f"Source #{i + 1}"):
                st.write(doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": full_response})


def show_sidebar():
    with st.sidebar:
        st.header("Document Sources")

        # Show connection managers in tabs
        tab1, tab2, tab3 = st.tabs(["Files", "FTP", "S3"])

        with tab1:
            source_type = st.radio(
                "Select source type",
                ["File Upload", "Local Directory"]
            )

            uploaded_files = None
            directory_path = None

            if source_type == "File Upload":
                uploaded_files = st.file_uploader(
                    label="Upload PDFs",
                    type=["pdf"],
                    accept_multiple_files=True
                )

            elif source_type == "Local Directory":
                directory_path = st.text_input("Directory path")

        with tab2:
            show_ftp_manager()

        with tab3:
            show_s3_manager()

        # Add refresh button
        if st.button("Refresh Documents", type="primary"):
            st.session_state.chain_initialized = False
            st.experimental_rerun()

        # Show active connections
        st.markdown("---")
        st.subheader("Active Connections")

        if st.session_state.ftp_connections:
            st.write("📁 FTP Servers:")
            for conn in st.session_state.ftp_connections:
                st.write(f"  • {conn['name']}")

        if st.session_state.s3_connections:
            st.write("🪣 S3 Buckets:")
            for conn in st.session_state.s3_connections:
                st.write(f"  • {conn['name']}")

        return uploaded_files, directory_path


def show_chat_section(chain):
    st.header("DocWain Chat")

    # Show message history in a container with fixed height
    with st.container():
        st.markdown("""
            <style>
                .chat-container {
                    height: 600px;
                    overflow-y: auto;
                    border: 1px solid #ddd;
                    padding: 1rem;
                    border-radius: 0.5rem;
                    margin-bottom: 1rem;
                }
            </style>
        """, unsafe_allow_html=True)

        with st.container():
            for message in st.session_state.messages:
                role = message["role"]
                avatar_path = (
                    Config.Path.IMAGES_DIR / "assistant-avatar.png"
                    if role == "assistant"
                    else Config.Path.IMAGES_DIR / "user-avatar.png"
                )
                with st.chat_message(role, avatar=str(avatar_path)):
                    st.markdown(message["content"])

    # Chat input at the bottom
    if prompt := st.chat_input("Ask your question here"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message(
                "user",
                avatar=str(Config.Path.IMAGES_DIR / "user-avatar.png"),
        ):
            st.markdown(prompt)
        asyncio.run(ask_chain(prompt, chain))


def initialize_chain(uploaded_files, directory_path):
    if (uploaded_files or directory_path or
            st.session_state.ftp_connections or
            st.session_state.s3_connections):
        with st.spinner("Analyzing documents from all sources..."):
            try:
                chain = build_qa_chain(
                    files=uploaded_files,
                    directory=directory_path
                )
                st.session_state.chain_initialized = True
                return chain
            except ValueError as e:
                st.error(str(e))
                st.session_state.chain_initialized = False
                return None
    else:
        st.info("Please provide at least one document source to start chatting!")
        st.session_state.chain_initialized = False
        return None


def main():
    st.set_page_config(
        page_title="DHS DocWain",
        page_icon=":mag:",
        layout="wide"
    )

    st.markdown("""
        <style>
            .st-emotion-cache-p4micv {
                width: 2.75rem;
                height: 2.75rem;
            }
            .stAlert {
                margin-top: 1rem;
            }
            .main-header {
                text-align: center;
                padding: 1rem 0;
                background: linear-gradient(90deg, #1E88E5 0%, #1565C0 100%);
                color: white;
                border-radius: 0.5rem;
                margin-bottom: 2rem;
            }
        </style>
    """, unsafe_allow_html=True)

    initialize_session_state()

    if 0 < Config.CONVERSATION_MESSAGES_LIMIT <= len(st.session_state.messages):
        st.warning(
            "You have reached the conversation limit. Refresh the page to start a new conversation."
        )
        st.stop()

    # Create layout
    st.markdown('<div class="main-header"><h1>DocWain</h1></div>', unsafe_allow_html=True)

    # Sidebar for document sources
    uploaded_files, directory_path = show_sidebar()

    # Main chat section
    if 'chain_initialized' not in st.session_state:
        st.session_state.chain_initialized = False

    chain = initialize_chain(uploaded_files, directory_path)

    if chain and st.session_state.chain_initialized:
        show_chat_section(chain)


if __name__ == "__main__":
    main()