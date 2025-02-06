import streamlit as st
import requests

BASE_URL = "http://localhost:8000/api/admin"

def show():
    st.markdown("<h1 style='text-align: center;'>🛠️ Admin Dashboard</h1>", unsafe_allow_html=True)

    if "auth_token" not in st.session_state:
        st.warning("🚨 Please login first!")
        return

    headers = {"Authorization": f"Bearer {st.session_state['auth_token']}"}

    # Fetch Pending Users
    st.subheader("👤 Pending User Approvals")
    users = requests.get(f"{BASE_URL}/users", headers=headers).json()
    for user in users:
        col1, col2, col3 = st.columns([3, 1, 1])
        col1.write(f"🧑‍💼 {user['username']} ({user['email']})")
        if col2.button(f"✅ Approve", key=f"approve_{user['id']}"):
            requests.post(f"{BASE_URL}/approve/{user['id']}", headers=headers)
            st.success(f"✅ User {user['username']} approved!")
        if col3.button(f"❌ Remove", key=f"remove_{user['id']}"):
            requests.post(f"{BASE_URL}/remove/{user['id']}", headers=headers)
            st.error(f"❌ User {user['username']} removed!")

    st.markdown("---")
    st.subheader("📁 Document Sources Management")
    source_type = st.selectbox("Select Source Type", ["AWS S3", "Azure Blob", "FTP", "Local"])
    source_path = st.text_input("Enter Path or Connection String")
    if st.button("➕ Add Document Source"):
        response = requests.post(f"{BASE_URL}/add_source", json={"type": source_type, "path": source_path}, headers=headers)
        if response.status_code == 200:
            st.success("✅ Document source added successfully!")
        else:
            st.error("⚠️ Error adding document source.")
