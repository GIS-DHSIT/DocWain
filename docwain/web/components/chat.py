import asyncio
import streamlit as st
from typing import Optional
from langchain.schema.runnable import Runnable
import os,sys
path = os.getcwd()
sys.path.append(path)
from docwain.utils.logger import setup_logger

logger = setup_logger()


def initialize_chat_state():
    """Initialize chat-related session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi! What would you like to know about your documents?",
            }
        ]
    if "current_response" not in st.session_state:
        st.session_state.current_response = ""
    if "processing" not in st.session_state:
        st.session_state.processing = False


async def get_response(chain: Runnable, question: str) -> tuple[str, list]:
    """Get response from the chain"""
    try:
        response = await chain.ainvoke({"question": question})
        # For now, return empty sources - you can extend this
        return response, []
    except Exception as e:
        logger.error(f"Error getting response: {e}")
        return f"Error: {str(e)}", []


def on_submit():
    """Handle form submission"""
    if st.session_state.question.strip():
        st.session_state.processing = True
        st.session_state.messages.append({
            "role": "user",
            "content": st.session_state.question
        })
        # Clear the input
        st.session_state.question = ""


def show_chat_section(chain: Optional[Runnable]):
    """Display the chat interface"""
    initialize_chat_state()

    st.header("Chat with your documents")

    if not chain:
        st.info("Please upload documents to start chatting.")
        return

    # Display chat messages
    messages_container = st.container()

    # Input form at the bottom
    with st.container():
        st.text_input(
            "Ask a question:",
            key="question",
            on_change=on_submit,
            placeholder="Type your question here...",
            label_visibility="collapsed"
        )

    # Display messages
    with messages_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # Process new message if needed
    if st.session_state.processing:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get latest user question
                user_q = st.session_state.messages[-1]["content"]

                try:
                    # Create new event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    # Get response
                    response, sources = loop.run_until_complete(get_response(chain, user_q))
                    loop.close()

                    # Add response to messages
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Error processing message: {e}")

                finally:
                    st.session_state.processing = False
                    st.rerun()