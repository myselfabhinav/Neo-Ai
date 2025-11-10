import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === API Keys ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# === Optional API keys for future integrations ===
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# === Validation ===
missing_keys = []
for key_name, key_value in [
    ("OPENAI_API_KEY", OPENAI_API_KEY),
    ("GROQ_API_KEY", GROQ_API_KEY),
    ("GEMINI_API_KEY", GEMINI_API_KEY),
    ("GOOGLE_API_KEY", GOOGLE_API_KEY),
    ("GOOGLE_CSE_ID", GOOGLE_CSE_ID),
]:
    if not key_value:
        missing_keys.append(key_name)

if missing_keys:
    print(f"[⚠️ Warning] Missing API keys in .env file: {', '.join(missing_keys)}")

# === Global Settings ===
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "data/vector_store")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# === Logging Configuration ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# === Helper Function ===
def print_config_summary():
    """Prints a summary of active configuration values."""
    print("\n===== 🔧 Configuration Summary =====")
    print(f"OpenAI Key Loaded: {'✅' if OPENAI_API_KEY else '❌'}")
    print(f"Groq Key Loaded: {'✅' if GROQ_API_KEY else '❌'}")
    print(f"Gemini Key Loaded: {'✅' if GEMINI_API_KEY else '❌'}")
    print(f"Google API Key Loaded: {'✅' if GOOGLE_API_KEY else '❌'}")
    print(f"Google CSE ID Loaded: {'✅' if GOOGLE_CSE_ID else '❌'}")
    print(f"Default Model: {DEFAULT_MODEL}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
