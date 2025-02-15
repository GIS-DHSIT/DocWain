import streamlit as st
import duckdb
import hashlib
import smtplib
import re
import pandas as pd
from email.message import EmailMessage
from email_validator import validate_email, EmailNotValidError

# Database Configuration
DB_FILE = "docUsers.duckdb"

# Email Configuration (Replace with Your Credentials)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SENDER = "muthu.g.subramanian@gmail.com"
EMAIL_PASSWORD = "mywx wprp keww afpf"

# Connect to DuckDB
conn = duckdb.connect(DB_FILE)
conn.execute("""
    CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        password_hash TEXT,
        role TEXT DEFAULT 'user',
        login_count INTEGER DEFAULT 0,
        last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")


# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Function to check if email exists
def email_exists(email):
    result = conn.execute("SELECT COUNT(*) FROM users WHERE email = ?", (email,)).fetchone()
    return result[0] > 0


def add_user(email, password, role="user"):
    password_hash = hash_password(password)
    conn.execute("INSERT INTO users VALUES (?, ?, ?, 0, CURRENT_TIMESTAMP)", (email, password_hash, role))


# Function to verify user login & update login details
def verify_user(email, password):
    password_hash = hash_password(password)
    result = conn.execute("SELECT role FROM users WHERE email = ? AND password_hash = ?",
                          (email, password_hash)).fetchone()

    if result:
        conn.execute("UPDATE users SET login_count = login_count + 1, last_login = CURRENT_TIMESTAMP WHERE email = ?",
                     (email,))
        return result[0]
    return None  # Returns role if user is found


# Function to get all users (for admin)
def get_all_users():
    return pd.DataFrame(conn.execute("SELECT email, role, login_count, last_login FROM users").fetchall(),
                        columns=["Email", "Role", "Login Count", "Last Login"])


# Function to delete a user
def delete_user(email):
    conn.execute("DELETE FROM users WHERE email = ?", (email,))


# Function to send a welcome email with login instructions
def send_welcome_email(email, password):
    try:
        msg = EmailMessage()
        msg["Subject"] = "Welcome - Your Account Details"
        msg["From"] = EMAIL_SENDER
        msg["To"] = email
        msg.set_content(f"""
        Hello,

        Welcome to the app! You have successfully registered.

        Your login details:
        - Email: {email}
        - Password: {password} (Please change it after logging in)

        Login at: http://yourapp.com/login

        Regards,
        Admin
        """)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)

        st.success(f"✅ Welcome email sent to {email}!")

    except Exception as e:
        st.error(f"❌ Email could not be sent: {e}")


# Toggle Pages
def switch_to(page):
    st.session_state["admin_page"] = page

def switch_to_signup():
    st.session_state["page"] = "Sign Up"

def switch_to_login():
    st.session_state["page"] = "Login"

def go_back_to_main():
    st.session_state["admin_page"] = "MainApp"

# Login Page
def login_page():
    st.sidebar.image(r"C:\Users\MuthuSubramanian\PycharmProjects\DocWain\images\Docwain-ai.gif")
    st.title("🔐 DocWain Login")

    col1, col2 = st.columns([2, 3])
    email = col1.text_input("Email", key="login_email", placeholder="Enter your email")
    password = col1.text_input("Password", type="password", key="login_password", placeholder="Enter your password")

    if col1.button("Login", use_container_width=True):
        role = verify_user(email, password)
        if role:
            st.session_state["authenticated"] = True
            st.session_state["email"] = email
            st.session_state["role"] = role
            st.success(f"✅ Welcome, {email} ({role})!")
            st.experimental_rerun()
        else:
            st.error("❌ Invalid email or password.")
    st.sidebar.markdown("## Join Us!")
    st.sidebar.button("Create Account", use_container_width=True, on_click=switch_to_signup)
    col2.image(r"C:\Users\MuthuSubramanian\PycharmProjects\DocWain\images\login.png",use_column_width=True)


# Sign-Up Page
def signup_page():
    st.title("📝 Sign Up")

    col1, col2 = st.columns([3, 3])
    new_email = col1.text_input("Email", key="signup_email", placeholder="you@example.com")
    new_password = col1.text_input("Password", type="password", key="signup_password",
                                   placeholder="Create a strong password")
    confirm_password = col1.text_input("Confirm Password", type="password", key="signup_confirm",
                                       placeholder="Re-enter password")

    if col1.button("Sign Up", use_container_width=True):
        if email_exists(new_email):
            st.error("❌ Email already registered.")
        elif new_password != confirm_password:
            st.error("❌ Passwords do not match.")
        else:
            add_user(new_email, new_password, role="user")
            st.success("✅ Account created! Check your email for login details.")
            st.experimental_rerun()

    col1.button("Back to Login", use_container_width=True, on_click=switch_to_login)
    col2.image(r"C:\Users\MuthuSubramanian\PycharmProjects\DocWain\images\signup.png")


def admin_dashboard():
    st.title("📊 Admin Dashboard")

    # Retrieve user stats
    users_df = get_all_users()
    total_users = len(users_df)
    most_active_user = users_df.sort_values(by="Login Count", ascending=False).iloc[0][
        "Email"] if total_users > 0 else "N/A"

    col1, col2 = st.columns(2)
    col1.metric("Total Users", total_users)
    col2.metric("Most Active User", most_active_user)

    st.write("### User Login Activity")
    st.dataframe(users_df)


# User Management
def user_management():
    st.title("👤 User Management")

    users_df = get_all_users()
    st.dataframe(users_df)

    st.write("### Delete User")
    user_to_delete = st.selectbox("Select User to Delete", users_df["Email"].tolist(), key="delete_user")

    if st.button("Delete User"):
        if user_to_delete == "muthu@dhsit.co.uk":
            st.error("❌ Cannot delete the admin account.")
        else:
            delete_user(user_to_delete)
            st.success(f"✅ User {user_to_delete} deleted successfully.")
            st.experimental_rerun()


# Settings Page
def admin_settings():
    st.title("⚙️ Document Settings")

    # Placeholder for future settings implementation
    st.write("🔧 Future: Add configuration settings here.")


# Main Application
def landingPage():
    if not email_exists("muthu@dhsit.co.uk"):
        add_user("muthu@dhsit.co.uk", "Forgetit*88", role="admin")

    st.sidebar.title("🔓 Logged In")
    st.sidebar.write(f"📧 Email: {st.session_state['email']} ({st.session_state['role']})")

    if "admin_page" not in st.session_state:
        st.session_state["admin_page"] = "Admin"

    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["email"] = None
        st.session_state["role"] = None
        st.experimental_rerun()

    st.title("🎉 Welcome to the DocWain!")


