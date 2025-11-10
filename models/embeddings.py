import sys
import os
import logging
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
except Exception:
    OpenAIEmbeddings = HuggingFaceEmbeddings = None

# Gemini removed for PDF uploads
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except Exception:
    GoogleGenerativeAIEmbeddings = None

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

VECTORSTORE_PATH = "data/vectorstore.bin"
DOCS_PATH = "data/docs"

class EmbeddingModelFallback:
    def __init__(self, dim: int = 384):
        self._dim = dim

    def get_embeddings(self, texts):
        import numpy as np
        logging.warning("⚠️ Using fallback embeddings (random vectors).")
        return np.random.rand(len(texts), self._dim).tolist()

    def get_embedding_dim(self):
        return self._dim

def get_embedding_model(provider: str = "openai"):
    provider = (provider or "openai").lower()
    try:
        if provider == "openai":
            if OPENAI_API_KEY and OpenAIEmbeddings:
                logging.info("✅ Using OpenAI embeddings.")
                return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            else:
                logging.warning("⚠️ OpenAI unavailable, falling back to HuggingFace or random.")
                if HuggingFaceEmbeddings:
                    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                return EmbeddingModelFallback()
        elif provider == "huggingface":
            if HuggingFaceEmbeddings:
                logging.info("✅ Using HuggingFace embeddings.")
                return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            logging.warning("⚠️ HuggingFace unavailable, using fallback.")
            return EmbeddingModelFallback()
        else:
            raise ValueError("Invalid provider. Use 'openai' or 'huggingface'.")
    except Exception:
        logging.exception("❌ Error creating embedding model; using fallback.")
        return EmbeddingModelFallback()

def get_free_embeddings():
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        logging.info(f"✅ Using free HuggingFace embeddings for RAG: {model_name}")
        return embeddings
    except Exception as e:
        logging.error(f"❌ Failed to load HuggingFace embeddings: {e}")
        return EmbeddingModelFallback()

def get_vectorstore(provider: str = "openai"):
    try:
        from models.vector_store import load_or_create_vectorstore
    except Exception as e:
        logging.warning(f"⚠️ Could not import models.vector_store. Error: {e}")
        load_or_create_vectorstore = None

    embedding_model = get_embedding_model(provider)

    if load_or_create_vectorstore:
        try:
            vs = load_or_create_vectorstore(
                embedding_model=embedding_model,
                docs_path=DOCS_PATH,
                vectorstore_path=VECTORSTORE_PATH
            )
            logging.info("✅ Vectorstore created/loaded successfully.")
            return vs
        except Exception:
            logging.exception("❌ Failed to load/create vectorstore, using stub fallback.")

    class VectorStoreStub:
        def __init__(self, embedding_model):
            self.embedding_model = embedding_model
            self._docs = []

        def add_documents(self, docs):
            self._docs.extend(docs)
            return True

        def search(self, query, top_k=3):
            results = []
            for d in self._docs:
                if query.lower() in d.lower():
                    results.append((d, 1.0))
            return results[:top_k]

    logging.warning("⚠️ Using VectorStoreStub (in-memory, not persistent).")
    return VectorStoreStub(embedding_model)

if __name__ == "__main__":
    print("Testing embeddings module...\n")
    vs_chat = get_vectorstore("openai")
    print("✅ Vectorstore (chat):", type(vs_chat))
    rag_embeddings = get_free_embeddings()
    print("✅ RAG embeddings:", type(rag_embeddings))
