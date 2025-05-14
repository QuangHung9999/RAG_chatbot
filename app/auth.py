import streamlit as st
import sqlite3
import os

# Ensure the database is in the project root or a dedicated data directory
# For simplicity, placing it in the project root for now.
DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "users.db")


def init_db():
    """Initializes the SQLite database and users table."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            username TEXT UNIQUE NOT NULL,
                            password TEXT NOT NULL)"""
        )
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error during initialization: {e}")
    finally:
        if conn:
            conn.close()


def add_user(username, password):
    """Adds a new user to the database."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)", (username, password)
        )
        conn.commit()
        st.success("Registration successful! You can now login.")
    except sqlite3.IntegrityError:
        st.error("Username already exists!")
    except sqlite3.Error as e:
        st.error(f"Database error during registration: {e}")
    finally:
        if conn:
            conn.close()


def validate_login(username, password):
    """Validates user login credentials."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE username = ? AND password = ?", (username, password)
        )
        user = cursor.fetchone()
        if user:
            return True
        return False
    except sqlite3.Error as e:
        st.error(f"Database error during login: {e}")
        return False
    finally:
        if conn:
            conn.close()


def login_page():
    """Displays the login page and handles login logic."""
    st.title("Login to RAG Chatbot")
    st.markdown("_(Using local SQLite database. For AWS deployment, this would be AWS Cognito.)_")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")

    if login_button:
        if validate_login(username, password):
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            # Do not call st.success here, it will be handled by rerun
            st.rerun()
        else:
            st.error("Invalid username or password.")


def register_page():
    """Displays the registration page and handles registration logic."""
    st.title("Register for RAG Chatbot")

    with st.form("register_form"):
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        register_button = st.form_submit_button("Register")

    if register_button:
        if new_username and new_password:
            add_user(new_username, new_password)
            # Success message is handled by add_user
        else:
            st.error("Please fill both username and password fields.")