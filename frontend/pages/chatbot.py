import streamlit as st
import requests
from streamlit_extras.chat_elements import message
from streamlit_extras.toggle_switch import st_toggle_switch

API_URL = "http://localhost:8000"


def chatbot_ui():
    st.title("🤖 AI Chatbot")

    # 🎨 Dark Mode Toggle
    dark_mode = st_toggle_switch("🌙 Dark Mode", default_value=False)

    # Chat History Storage
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 🗨️ Chat Container
    chat_container = st.container()

    # 📜 Display Chat History
    for chat in st.session_state.chat_history:
        message(chat["text"], is_user=chat["is_user"], key=chat["id"])

    # ⌨️ Chat Input
    user_query = st.chat_input("Type your question...")

    if user_query:
        st.session_state.chat_history.append(
            {"text": user_query, "is_user": True, "id": f"user-{len(st.session_state.chat_history)}"})
        message(user_query, is_user=True, key=f"user-{len(st.session_state.chat_history)}")

        # API Call
        with st.spinner("Thinking... 💭"):
            response = requests.get(f"{API_URL}/query/", params={"question": user_query})
            answer = response.json().get("answer", "Sorry, I couldn't find an answer.")

        st.session_state.chat_history.append(
            {"text": answer, "is_user": False, "id": f"bot-{len(st.session_state.chat_history)}"})
        message(answer, is_user=False, key=f"bot-{len(st.session_state.chat_history)}")

    # 📜 Collapsible Chat History
    with st.expander("📜 Chat Logs"):
        history_response = requests.get(f"{API_URL}/history/").json()["history"]
        for chat in history_response:
            st.write(f"**Q:** {chat[1]}")
            st.write(f"**A:** {chat[2]}")
            st.markdown("---")
