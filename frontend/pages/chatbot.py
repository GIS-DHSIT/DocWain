import streamlit as st
import requests
import time

BASE_URL = "http://localhost:8000/api/chat"

def show():
    st.markdown("<h1 style='text-align: center;'>🤖 Chat with DocWain</h1>", unsafe_allow_html=True)

    if "auth_token" not in st.session_state:
        st.warning("🚨 Please login first!")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    query = st.text_input("💬 Ask a question about your documents")
    response_type = st.radio("📖 Response Type", ["Short", "Detailed"], horizontal=True)

    if st.button("🔍 Ask DocWain"):
        with st.spinner("🧠 Thinking..."):
            headers = {"Authorization": f"Bearer {st.session_state['auth_token']}"}
            params = {"query": query, "response_type": response_type.lower()}
            response = requests.get(BASE_URL, params=params, headers=headers)

            if response.status_code == 200:
                data = response.json()
                st.session_state.messages.append({"query": query, "response": data["response"], "sources": data["source_documents"]})
            else:
                st.error("⚠️ Error retrieving response.")

    # Display Chat History with Typing Effect
    for msg in reversed(st.session_state.messages):
        st.markdown(f"**🧑‍💻 You:** {msg['query']}")
        with st.expander("🤖 DocWain's Answer"):
            st.write(msg["response"])
            st.info(f"📄 Sources: {', '.join(msg['sources'])}")
        time.sleep(0.3)  # Simulated Typing Effect
