import os
import dotenv

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

# Application Configuration
APP_ENV = os.getenv("APP_ENV", "development")

# API Keys 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Database Configuration
DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "users.db"))
CHAT_HISTORY_DB_PATH = os.getenv("CHAT_HISTORY_DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history.db"))

# Models Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
GOOGLE_MODELS = os.getenv("GOOGLE_MODELS", "gemini-1.5-flash").split(",")
DEFAULT_GOOGLE_MODEL = os.getenv("DEFAULT_GOOGLE_MODEL", "gemini-1.5-flash")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.35"))

# File paths and directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS_DIR = os.path.join(PROJECT_ROOT, os.getenv("DOCUMENTS_DIR", "documents"))
SOURCE_DOCUMENT_FILENAME = os.getenv("SOURCE_DOCUMENT_FILENAME", "Company-10k-18pages.pdf")
SOURCE_DOCUMENT_PATH = os.path.join(DOCUMENTS_DIR, SOURCE_DOCUMENT_FILENAME)
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, os.getenv("VECTOR_STORE_DIR", "vector_store_data"))
VECTOR_STORE_INDEX_NAME = os.getenv("VECTOR_STORE_INDEX_NAME", "faiss_index")
VECTOR_STORE_PATH = os.path.join(VECTOR_STORE_DIR, VECTOR_STORE_INDEX_NAME)

# UI Configuration 
APP_TITLE = os.getenv("APP_TITLE", "DocuMentor RAG")
APP_SUBTITLE = os.getenv("APP_SUBTITLE", "Chat with your PDF documents intelligently!")
APP_ICON = os.getenv("APP_ICON", "📚")

# RAG Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "10")) 
