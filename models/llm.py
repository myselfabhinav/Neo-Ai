import os
import logging
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)

try:
    load_dotenv()
    logging.info("✅ .env loaded successfully.")
except Exception as e:
    logging.warning(f"⚠️ Failed to load .env file: {e}")

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None
    logging.warning("⚠️ langchain_openai not available.")

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None
    logging.warning("⚠️ langchain_groq not available.")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None
    logging.warning("⚠️ langchain_google_genai not available.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

logging.info(f"OPENAI_API_KEY loaded: {'Yes' if OPENAI_API_KEY else 'No'}")
logging.info(f"GROQ_API_KEY loaded: {'Yes' if GROQ_API_KEY else 'No'}")
logging.info(f"GEMINI_API_KEY loaded: {'Yes' if GEMINI_API_KEY else 'No'}")

def get_openai_model(model_name="gpt-4o-mini", temperature=0.6):
    if not ChatOpenAI or not OPENAI_API_KEY:
        return "openai-model-stub"
    try:
        model = ChatOpenAI(
            model=model_name,
            api_key=OPENAI_API_KEY,
            temperature=temperature,
            max_tokens=2048
        )
        logging.info(f"✅ OpenAI model initialized: {model_name}")
        return model
    except Exception as e:
        logging.error(f"❌ Failed to initialize OpenAI: {e}")
        return "openai-model-stub"

def get_chatgroq_model(model_name="groq/llama-3.1-70b", temperature=0.6):
    if not ChatGroq or not GROQ_API_KEY:
        return "groq-model-stub"
    try:
        model = ChatGroq(
            model=model_name,
            api_key=GROQ_API_KEY,
            temperature=temperature,
            max_tokens=2048
        )
        logging.info(f"✅ Groq model initialized: {model_name}")
        return model
    except Exception as e:
        logging.error(f"❌ Failed to initialize Groq: {e}")
        return "groq-model-stub"

def get_gemini_model(model_name="models/gemini-2.0-flash", temperature=0.6):
    if not ChatGoogleGenerativeAI or not GEMINI_API_KEY:
        return "gemini-model-stub"
    try:
        model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=GEMINI_API_KEY,
            temperature=temperature,
            max_output_tokens=2048
        )
        logging.info(f"✅ Gemini model initialized: {model_name}")
        return model
    except Exception as e:
        logging.error(f"❌ Failed to initialize Gemini: {e}")
        return "gemini-model-stub"

def get_free_embeddings():
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        logging.info(f"✅ Using free HuggingFace embeddings: {model_name}")
        return embeddings
    except Exception as e:
        logging.error(f"❌ Failed to load HuggingFace embeddings: {e}")
        return "huggingface-embeddings-stub"

def invoke_model(model, messages, provider="OpenAI", system_prompt=""):
    if isinstance(model, str) and model.endswith("-stub"):
        return f"⚠️ Stub response for {provider} (model not available)"
    try:
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    except ImportError:
        return f"⚠️ langchain_core not installed for {provider}"
    prepared_messages = [SystemMessage(content=system_prompt)] if system_prompt else []
    for msg in messages:
        if msg["role"] == "user":
            prepared_messages.append(HumanMessage(content=msg["content"]))
        else:
            prepared_messages.append(AIMessage(content=msg["content"]))
    try:
        response = model.invoke(prepared_messages)
        if hasattr(response, "content"):
            return str(response.content).strip()
        return str(response).strip()
    except Exception as e:
        logging.error(f"⚠️ {provider} invocation error: {e}")
        return f"⚠️ Error from {provider}: {str(e)}"

def get_best_available_model():
    model = get_openai_model()
    if not isinstance(model, str) or not model.endswith("-stub"):
        return model, "OpenAI"
    model = get_chatgroq_model()
    if not isinstance(model, str) or not model.endswith("-stub"):
        return model, "Groq"
    model = get_gemini_model()
    if not isinstance(model, str) or not model.endswith("-stub"):
        return model, "Gemini"
    return "no-model-stub", "None"

if __name__ == "__main__":
    print("🔑 OPENAI_API_KEY:", bool(OPENAI_API_KEY))
    print("🔑 GROQ_API_KEY:", bool(GROQ_API_KEY))
    print("🔑 GEMINI_API_KEY:", bool(GEMINI_API_KEY))
    model, provider = get_best_available_model()
    print(f"✅ Using model provider: {provider}")
    emb = get_free_embeddings()
    print(f"✅ Embeddings loaded: {type(emb)}")
