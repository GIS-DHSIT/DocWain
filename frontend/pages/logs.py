import streamlit as st
import requests

API_URL = "http://localhost:8000"

def logs_ui():
    st.title("📜 System & Chat Logs")

    # Fetch Chat History
    with st.expander("🗨️ Chat Logs"):
        chat_logs = requests.get(f"{API_URL}/history/").json()["history"]
        for chat in chat_logs:
            st.write(f"**Q:** {chat[1]}")
            st.write(f"**A:** {chat[2]}")
            st.markdown("---")

    # Fetch System Logs (Future Feature)
    st.subheader("⚙️ System Logs (Coming Soon...)")
