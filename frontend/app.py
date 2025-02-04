import streamlit as st

# ---- 🎨 UI Config ----
st.set_page_config(
    page_title="Docwain AI",
    page_icon="📄",
    layout="wide",
)

# ---- 🏠 Sidebar Navigation ----
# st.sidebar.image("assets/logo.png", width=200)
st.sidebar.title("📚 Docwain AI")
page = st.sidebar.radio("📌 Navigation", ["Chatbot", "Admin Panel", "Logs"])

if page == "Chatbot":
    from pages.chatbot import chatbot_ui
    chatbot_ui()

elif page == "Admin Panel":
    from pages.admin import admin_ui
    admin_ui()

elif page == "Logs":
    from pages.logs import logs_ui
    logs_ui()
