import streamlit as st
import requests

API_URL = "http://localhost:8000"

def admin_ui():
    st.title("🛠️ Admin Panel - Document Management & Configuration")

    # 🎯 Tabs for better organization
    tab1, tab2 = st.tabs(["📂 Upload & Manage Documents", "⚙️ System Configuration"])

    # 📂 File Upload Section
    with tab1:
        uploaded_file = st.file_uploader("Upload Document", type=["pdf", "docx", "xlsx", "csv", "ppt"])
        if st.button("Upload"):
            if uploaded_file:
                res = requests.post(f"{API_URL}/upload/", files={"file": uploaded_file.getvalue()})
                st.success(res.json().get("status"))

        st.subheader("📄 Uploaded Documents")
        docs = requests.get(f"{API_URL}/history/").json()["history"]
        for doc in docs:
            st.write(f"📂 {doc[1]} - `{doc[2]}`")

    # ⚙️ Configuration Section
    with tab2:
        st.subheader("🔧 Configure System Settings")

        aws_key = st.text_input("AWS Access Key", value="your-access-key")
        aws_secret = st.text_input("AWS Secret Key", value="your-secret-key")
        s3_bucket = st.text_input("S3 Bucket Name", value="your-bucket-name")

        ftp_server = st.text_input("FTP Server", value="ftp.example.com")
        ftp_user = st.text_input("FTP Username", value="your-ftp-user")
        ftp_pass = st.text_input("FTP Password", value="your-ftp-password")

        openai_key = st.text_input("OpenAI API Key", value="your-openai-key")

        if st.button("Save Changes"):
            updated_config = {
                "AWS_ACCESS_KEY": aws_key,
                "AWS_SECRET_KEY": aws_secret,
                "S3_BUCKET_NAME": s3_bucket,
                "FTP_SERVER": ftp_server,
                "FTP_USERNAME": ftp_user,
                "FTP_PASSWORD": ftp_pass,
                "OPENAI_API_KEY": openai_key
            }
            for key, value in updated_config.items():
                requests.post(f"{API_URL}/config/update/", json={"key": key, "value": value})
            st.success("✅ Configuration Updated!")
