import streamlit as st
import importlib
from streamlit_option_menu import option_menu

# Set up page configuration
st.set_page_config(page_title="DocWain AI", layout="wide", page_icon="📄")

# Define available pages dynamically
PAGES = {
    "Landing Page": "pages.landing",
    "Login": "pages.login",
    "Signup": "pages.signup",
    "Chatbot": "pages.chatbot",
    "Admin Dashboard": "pages.admin",
    "Document Viewer": "pages.document_viewer",
}

# Initialize session state for navigation
if "current_page" not in st.session_state:
    st.session_state.current_page = "Landing Page"

# Sidebar for navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3178/3178286.png", width=80)
    st.title("📄 DocWain AI")
    st.markdown("🚀 AI-powered document chatbot")

    # Navigation Menu
    selected_page = option_menu(
        menu_title="Navigation",
        options=list(PAGES.keys()),
        icons=["house", "login", "person-plus", "robot", "gear", "folder"],
        default_index=list(PAGES.keys()).index(st.session_state.current_page)
    )

    # Update session state
    st.session_state.current_page = selected_page

# Dynamically load selected page
module = importlib.import_module(PAGES[st.session_state.current_page])
module.show()
