# import streamlit as st
# import requests
#
# BASE_URL = "http://localhost:8000/api/admin"
#
# def show():
#     st.title("📄 Document Viewer & Tagging")
#
#     if "auth_token" not in st.session_state:
#         st.warning("Please login first!")
#         return
#
#     headers = {"Authorization": f"Bearer {st.session_state['auth_token']}"}
#
#     # Fetch Documents
#     st.subheader("Uploaded Documents")
#     documents = requests.get(f"{BASE_URL}/documents", headers=headers).json()
#     for doc in documents:
#         st.write(f"📄 {doc['filename']} - Source: {doc['source']}")
#         tag = st.text_input(f"Tag for {doc['filename']}", key=doc["id"])
#         if st.button(f"Save Tag {doc['filename']}", key=doc["id"]):
#             requests.post(f"{BASE_URL}/tag_document", json={"doc_id": doc["id"], "tag": tag}, headers=headers)
#             st.success(f"Tagged {doc['filename']} as {tag}")

import streamlit as st
import requests

BASE_URL = "http://localhost:8000/api/documents"


def show():
    st.markdown("<h1 style='text-align: center;'>📄 Document Viewer</h1>", unsafe_allow_html=True)

    if "auth_token" not in st.session_state:
        st.warning("🚨 Please login first!")
        return

    headers = {"Authorization": f"Bearer {st.session_state['auth_token']}"}

    # Fetch Documents
    st.subheader("📁 Uploaded Documents")
    documents = requests.get(f"{BASE_URL}/list", headers=headers).json()

    search_query = st.text_input("🔍 Search Documents")
    filtered_docs = [doc for doc in documents if search_query.lower() in doc["filename"].lower()]

    for doc in filtered_docs:
        col1, col2, col3 = st.columns([3, 1, 1])
        col1.write(f"📄 {doc['filename']} - **Source:** {doc['source']}")
        tag = col2.text_input(f"Tag {doc['filename']}", key=doc["id"])
        if col3.button(f"✅ Save Tag", key=f"tag_{doc['id']}"):
            requests.post(f"{BASE_URL}/tag", json={"doc_id": doc["id"], "tag": tag}, headers=headers)
            st.success(f"✅ Tagged {doc['filename']} as {tag}")

        if st.button(f"📥 Download {doc['filename']}", key=f"download_{doc['id']}"):
            download_url = f"{BASE_URL}/download/{doc['id']}"
            st.markdown(f"[⬇️ Click to Download]({download_url})", unsafe_allow_html=True)
