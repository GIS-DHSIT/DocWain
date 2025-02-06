import streamlit as st
import requests
import streamlit.components.v1 as components

BASE_URL = "http://localhost:8000/api/auth"


def show():
    st.markdown("<h1 style='text-align: center;'>🔐 Login to DocWain</h1>", unsafe_allow_html=True)

    with st.form("login_form"):
        email = st.text_input("📧 Email", placeholder="Enter your email")
        password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
        submit_button = st.form_submit_button("Login")

    if submit_button:
        response = requests.post(f"{BASE_URL}/login", data={"username": email, "password": password})
        if response.status_code == 200:
            st.success("✅ Logged in successfully!")
            st.session_state["auth_token"] = response.json()["access_token"]
            st.balloons()
        else:
            st.error("❌ Invalid credentials, please try again!")
