import asyncio
import random

import streamlit as st
from dotenv import load_dotenv
import boto3
from azure.storage.blob import BlobServiceClient
from ftplib import FTP


from src.chain import ask_question, create_chain
from src.config import Config
from src.ingestor import Ingestor
from src.model import create_llm
from src.retriever import create_retriever
from src.uploader import upload_files
from src.src_conn.s3Connector import extracts3
from src.src_conn.azureBlob import azBlob
from src.src_conn.ftpConnector import ftpConn
from src.ui.login_signup import switch_to,landingPage,admin_settings,admin_dashboard,user_management
from src.ui.login_signup import login_page,signup_page,go_back_to_main
from api import run

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


@st.cache_resource(show_spinner=False)
def build_qa_chain(files):
    file_paths = upload_files(files)
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
            with st.expander(f"Source #{i+1}"):
                st.write(doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": full_response})


def show_upload_documents():
    holder = st.empty()
    with holder.container():
        st.header("DocWain")
        st.subheader("Document Wise-ai Nanobot ")
        uploaded_files = st.file_uploader(
            label="Upload files", type=["pdf","xlsx","csv","ppt","docx"], accept_multiple_files=True
        )
    if not uploaded_files:
        st.warning("Please upload PDF documents to continue!")
        st.stop()

    with st.spinner("Analyzing your document(s)..."):
        holder.empty()
        return build_qa_chain(uploaded_files)


def show_message_history():
    for message in st.session_state.messages:
        role = message["role"]
        avatar_path = (
            Config.Path.IMAGES_DIR / "assistant-avatar.png"
            if role == "assistant"
            else Config.Path.IMAGES_DIR / "user-avatar.png"
        )
        with st.chat_message(role, avatar=str(avatar_path)):
            st.markdown(message["content"])


def show_chat_input(chain):
    if prompt := st.chat_input("Ask your question here"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message(
            "user",
            avatar=str(Config.Path.IMAGES_DIR / "user-avatar.png"),
        ):
            st.markdown(prompt)
        asyncio.run(ask_chain(prompt, chain))


st.set_page_config(page_title="DHS DocWain", page_icon=":mag:")

st.html(
    """
<style>
    .st-emotion-cache-p4micv {
        width: 2.75rem;
        height: 2.75rem;
    }
</style>
"""
)
# Toggle Between Login & Sign-Up
if "page" not in st.session_state:
    st.session_state["page"] = "Login"

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["username"] = None
    st.session_state["role"] = None


# Show appropriate page
if st.session_state["authenticated"]:

    if st.session_state["role"] == "admin":
        st.sidebar.subheader("Admin Console")
        if st.sidebar.button("Dashboard"):
            switch_to("Dashboard")
        if st.sidebar.button("User Management"):
            switch_to("User Management")
        if st.sidebar.button("Document Settings"):
            switch_to("Settings")
        if st.session_state["admin_page"] == "Dashboard":
            admin_dashboard()
        if st.sidebar.button("🔙 Go Back to Main App"):
            go_back_to_main()
        elif st.session_state["admin_page"] == "User Management":
            user_management()
        elif st.session_state["admin_page"] == "Settings":
            admin_settings()
        elif st.session_state["admin_page"] == "Admin":
            landingPage()
    else:
        landingPage()

        sources = ['Local','FTP','AWS S3','Azure Blob','Others']
        src = st.sidebar.radio("Select Source",sources)
        if src == 'Local':
            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": "Hi! What do you want to know about your documents?",
                    }
                ]

            if 0 < Config.CONVERSATION_MESSAGES_LIMIT <= len(st.session_state.messages):
                st.warning(
                    "You have reached the conversation limit. Refresh the page to start a new conversation."
                )
                st.stop()

            chain = show_upload_documents()
            show_message_history()
            show_chat_input(chain)

        if src == 'FTP':
            with st.form("FTP Connection"):
                FTP_HOST = st.secrets["FTP_HOST"] if "FTP_HOST" in st.secrets else st.text_input("FTP Server")
                FTP_USER = st.secrets["FTP_USER"] if "FTP_USER" in st.secrets else st.text_input("Username")
                FTP_PASS = st.secrets["FTP_PASS"] if "FTP_PASS" in st.secrets else st.text_input("Password", type="password")
                FTP_DIR = st.secrets["FTP_DIR"] if "FTP_DIR" in st.secrets else st.text_input("directory")
                submitted = st.form_submit_button("Submit")
                if submitted:
                    # Connect to FTP Server
                    @st.cache_resource
                    def get_ftp_connection():
                        ftp = FTP(FTP_HOST)
                        ftp.login(FTP_USER, FTP_PASS)
                        ftp.cwd(FTP_DIR)
                        return ftp

                    ftp = get_ftp_connection()
                    # List files in the FTP directory
                    @st.cache_resource
                    def list_files():
                        return ftp.nlst()  # List file names

                    file_list = list_files()
                    selected_files = st.multiselect("Select Files to Extract Content:", file_list)
                    ftpConn(selected_files,ftp)

        if src == 'AWS S3':
            # st.columns(2,2,1)
            # AWS Configuration (Use environment variables or Streamlit secrets)
            AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY"] if "AWS_ACCESS_KEY" in st.secrets else st.text_input("YOUR_ACCESS_KEY",type="password")
            AWS_SECRET_KEY = st.secrets["AWS_SECRET_KEY"] if "AWS_SECRET_KEY" in st.secrets else st.text_input("YOUR_SECRET_KEY",type="password")
            AWS_REGION = st.text_input("Enter your Region",'eu-west-1')  # Change to your AWS region
            if AWS_REGION and AWS_SECRET_KEY and AWS_ACCESS_KEY:
                @st.cache_resource
                def get_s3_client():
                    return boto3.client(
                        "s3",
                        aws_access_key_id=AWS_ACCESS_KEY,
                        aws_secret_access_key=AWS_SECRET_KEY,
                        region_name=AWS_REGION,
                    )

                s3 = get_s3_client()
                extracts3(s3)

        if src =='Azure Blob':
            # Azure Storage Configuration
            AZURE_CONNECTION_STRING = st.secrets[
                "AZURE_CONNECTION_STRING"] if "AZURE_CONNECTION_STRING" in st.secrets else st.text_input("YOUR_AZURE_CONNECTION_STRING",type='password')
            CONTAINER_NAME = st.secrets[
                "AZURE_CONTAINER_NAME"] if "AZURE_CONTAINER_NAME" in st.secrets else st.text_input("YOUR_CONTAINER_NAME",type='password')

            if AZURE_CONNECTION_STRING and CONTAINER_NAME:
                @st.cache_resource
                def get_blob_service_client():
                    return BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)


                blob_service_client = get_blob_service_client()
                print(blob_service_client)
                container_client = blob_service_client.get_container_client(CONTAINER_NAME)

                # List files in the container
                @st.cache_resource
                def list_files():
                    return [blob.name for blob in container_client.list_blobs()]

                file_list = list_files()
                selected_files = st.multiselect("Select Files to Extract Content:", file_list)
                azBlob(selected_files,container_client)
        # run()
        # uvicorn.run(app, host="0.0.0.0", port=80)

else:
    if st.session_state["page"] == "Login":
        login_page()
    else:
        signup_page()