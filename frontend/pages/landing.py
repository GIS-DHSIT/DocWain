import streamlit as st

def show():
    st.markdown("<h1 style='text-align: center;'>🚀 Welcome to DocWain AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Your AI-powered document assistant</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔑 Login"):
            st.session_state.current_page = "Login"
            st.experimental_rerun()

    with col2:
        if st.button("📝 Signup"):
            st.session_state.current_page = "Signup"
            st.experimental_rerun()

    with col3:
        if st.button("🤖 Chatbot"):
            st.session_state.current_page = "Chatbot"
            st.experimental_rerun()

    st.markdown("---")
    st.subheader("📜 What You Can Do:")
    st.write("- **AI-powered document search and chat**")
    st.write("- **Upload & manage documents**")
    st.write("- **Admin dashboard for user management**")
    st.write("- **Secure login and user authentication**")
