import streamlit as st
import requests

BASE_URL = "http://localhost:8000/api/auth"

def show():
    st.markdown("<h1 style='text-align: center;'>📝 Create an Account</h1>", unsafe_allow_html=True)

    with st.form("signup_form"):
        username = st.text_input("👤 Username", placeholder="Enter your name")
        email = st.text_input("📧 Email", placeholder="Enter your email")
        password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
        confirm_password = st.text_input("🔑 Confirm Password", type="password", placeholder="Re-enter your password")
        domain = st.selectbox("🌍 Select Your Domain", ["Finance", "Operations", "Tech", "Business", "Misc", "All"])
        submit_button = st.form_submit_button("Sign Up")

    if submit_button:
        if password != confirm_password:
            st.error("❌ Passwords do not match!")
            return

        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "username": username,
            "email": email,
            "password": password,
            "domain": domain
        }

        response = requests.post(f"{BASE_URL}/signup", data=data, headers=headers)

        if response.status_code == 200:
            st.success("✅ Account created successfully! You can now login.")
            st.balloons()
        else:
            st.error("⚠️ Signup failed! Email may already be registered.")
