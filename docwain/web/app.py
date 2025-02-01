import os

# Configure TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import asyncio
import streamlit as st
from pathlib import Path
from typing import Optional, Any
import os,sys
path = os.getcwd()
sys.path.append(path)
from docwain.core.chain import create_chain
from docwain.core.ingestor import Ingestor
from docwain.core.processor import create_llm
from docwain.core.retriever import create_retriever
from docwain.utils.logger import setup_logger
from docwain.web.components.upload import show_upload_section
from docwain.web.components.chat import show_chat_section

logger = setup_logger()


def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi! What would you like to know about your documents?",
            }
        ]
    if "chat_chain" not in st.session_state:
        st.session_state.chat_chain = None


async def process_documents(files: Any, progress_callback: Any) -> Optional[Any]:
    """Process documents with progress tracking"""
    try:
        # Initialize progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(current: int, total: int, message: str):
            """Update progress bar and status message"""
            if total > 0:
                progress = float(current) / float(total)
                progress_bar.progress(progress)
                status_text.text(f"{message} ({current}/{total})")
            else:
                status_text.text(message)

        # Create ingestor with progress callback
        ingestor = Ingestor(progress_callback=update_progress)

        # Process documents
        with st.spinner("Processing documents..."):
            vector_store = await ingestor.ingest(files)
            llm = create_llm()
            retriever = create_retriever(llm, vector_store=vector_store)
            chain = create_chain(llm, retriever)

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        st.success("Documents processed successfully!")

        return chain

    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        st.error(f"Error processing documents: {str(e)}")
        return None


def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="DocWain",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_session_state()

    with st.sidebar:
        st.title("DocWain")
        st.markdown("---")

        # Upload section
        uploaded_files = show_upload_section()

        if uploaded_files:
            # Get event loop or create new one
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Process documents
            st.session_state.chat_chain = loop.run_until_complete(
                process_documents(uploaded_files, None)
            )

    # Main chat interface
    show_chat_section(st.session_state.chat_chain)


if __name__ == "__main__":
    main()