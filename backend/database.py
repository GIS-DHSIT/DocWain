import duckdb

DB_PATH = "docwain.db"
conn = duckdb.connect(DB_PATH)

# Initialize Config Table
def initialize_db():
    conn.execute("""
    CREATE TABLE IF NOT EXISTS config (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """)

    default_settings = {
        "UPLOAD_DIR": "data/",
        "MODELS_DIR": "models/",
        "INDEX_PATH": "models/index.faiss",
        "EMBEDDING_DIMENSIONS": "768",
        "LLM_MODEL": "gpt-3.5-turbo",
        "AWS_ACCESS_KEY": "your-access-key",
        "AWS_SECRET_KEY": "your-secret-key",
        "S3_BUCKET_NAME": "your-bucket-name",
        "FTP_SERVER": "ftp.example.com",
        "FTP_USERNAME": "your-ftp-user",
        "FTP_PASSWORD": "your-ftp-password",
        "OPENAI_API_KEY": "your-openai-key"
    }

    for key, value in default_settings.items():
        conn.execute("INSERT OR IGNORE INTO config (key, value) VALUES (?, ?)", (key, value))

    print("✅ Database Initialized!")

initialize_db()

# Functions to Get and Update Config
def get_config(key):
    result = conn.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
    return result[0] if result else None

def set_config(key, value):
    conn.execute("INSERT INTO config (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = ?", (key, value, value))

def get_all_config():
    return dict(conn.execute("SELECT key, value FROM config").fetchall())
