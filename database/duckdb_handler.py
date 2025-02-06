import duckdb
import uuid

# Connect to DuckDB
DATABASE_FILE = "documents.duckdb"
# Singleton Pattern: Use a Single Connection
class DuckDBConnection:
    _instance = None

    @staticmethod
    def get_connection():
        if DuckDBConnection._instance is None:
            DuckDBConnection._instance = duckdb.connect(DATABASE_FILE, read_only=False)
        return DuckDBConnection._instance

# Create Users Table (Fix UUID Issue)
conn = DuckDBConnection.get_connection()

conn.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,  -- Use TEXT instead of UUID
        username TEXT,
        email TEXT UNIQUE,
        password TEXT,
        domain TEXT,
        is_approved BOOLEAN DEFAULT FALSE
    )
""")

# Create Documents Table (Fix UUID Issue)
conn.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,  -- Use TEXT instead of UUID
        filename TEXT,
        content TEXT,
        source_type TEXT,
        tags TEXT
    )
""")

# Function to Generate UUID in Python
def generate_uuid():
    return str(uuid.uuid4())

# Create New User
def create_user(username, email, password, domain):
        user_id = str(uuid.uuid4())
        conn.execute("INSERT INTO users (id, username, email, password, domain) VALUES (?, ?, ?, ?, ?)",
                     (user_id, username, email, password, domain))
        return user_id
    # conn.execute("INSERT INTO users (username, email, password, domain) VALUES (?, ?, ?, ?)",
    #              (username, email, password, domain))
    # return conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()[0]

# Fetch User by Email
def get_user(email):
    result = conn.execute("SELECT id, username, email, password, domain, is_approved FROM users WHERE email=?",
                          (email,)).fetchone()
    return {"id": result[0], "username": result[1], "email": result[2], "password": result[3], "domain": result[4],
            "is_approved": result[5]} if result else None

# Get Pending Users
def get_pending_users():
    users = conn.execute("SELECT id, username, email FROM users WHERE is_approved = FALSE").fetchall()
    return [{"id": user[0], "username": user[1], "email": user[2]} for user in users]

# Approve User
def approve_user(user_id):
    updated = conn.execute("UPDATE users SET is_approved = TRUE WHERE id = ?", (user_id,))
    return updated.rowcount > 0

# Fetch All Documents
def get_documents():
    with duckdb.connect(DATABASE_FILE, read_only=False) as conn:
        return conn.execute("SELECT * FROM documents").fetchall()

# Tag Document
def tag_document(doc_id, tag):
    updated = conn.execute("UPDATE documents SET tags = ? WHERE id = ?", (tag, doc_id))
    return updated.rowcount > 0

# Function to Store Document Metadata
def store_document_metadata(filename, text, source_type):
    with duckdb.connect(DATABASE_FILE, read_only=False) as conn:
        conn.execute("INSERT INTO documents (id, filename, content, source_type) VALUES (?, ?, ?, ?)",
                     (str(uuid.uuid4()), filename, text, source_type))

def list_documents():
    docs = conn.execute("SELECT id, filename, tags FROM documents").fetchall()
    return [{"id": doc[0], "filename": doc[1], "tags": doc[2]} for doc in docs]

# # ✅ Tag a Document
# def tag_document(doc_id, tag):
#     conn.execute("UPDATE documents SET tags=? WHERE id=?", (tag, doc_id))
#     return True

# ✅ Get Document Path
def get_document_path(doc_id):
    result = conn.execute("SELECT filepath FROM documents WHERE id=?", (doc_id,)).fetchone()
    return result[0] if result else None
# Function to Fetch Documents
# def get_documents():
#     return conn.execute("SELECT * FROM documents").fetchall()