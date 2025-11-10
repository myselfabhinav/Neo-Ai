import os
import logging
from typing import List

from models.vector_store import EmbeddingModel, VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    RAG (Retrieval-Augmented Generation) pipeline for the Healthcare Chatbot.
    Retrieves relevant information from stored medical documents
    and augments LLM responses with factual context.
    """

    def __init__(self, docs_path: str = "data/medical_docs.txt"):
        """
        Initialize the RAG pipeline.
        Args:
            docs_path: Path to the text file containing medical documents.
        """
        try:
            self.docs_path = docs_path
            self.embedding_model = EmbeddingModel()
            self.vector_store = VectorStore(self.embedding_model)

            if os.path.exists(self.docs_path):
                self._load_documents_to_store()
            else:
                logger.warning(f"No document file found at {self.docs_path}. Add documents to improve retrieval.")
        except Exception as e:
            logger.error(f"Error initializing RAG pipeline: {e}")

    def _load_documents_to_store(self):
        """Load and embed documents from file if not already in vector store."""
        try:
            with open(self.docs_path, "r", encoding="utf-8") as f:
                docs = [line.strip() for line in f.readlines() if line.strip()]

            if len(self.vector_store.metadata) == 0:
                logger.info(f"Adding {len(docs)} medical documents to vector store...")
                self.vector_store.add_documents(docs)
                logger.info("Document embeddings created and stored successfully.")
            else:
                logger.info("Vector store already contains documents.")
        except Exception as e:
            logger.error(f"Error loading documents into vector store: {e}")

    def retrieve_relevant_docs(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve top-k relevant documents based on query."""
        try:
            results = self.vector_store.search(query, top_k=top_k)
            return [doc for doc, _ in results]
        except Exception as e:
            logger.error(f"Error retrieving relevant documents: {e}")
            return []

    def build_augmented_prompt(self, query: str) -> str:
        """Combine query and retrieved medical context into a single augmented prompt."""
        try:
            retrieved_docs = self.retrieve_relevant_docs(query, top_k=3)
            context = "\n".join(retrieved_docs) if retrieved_docs else "No relevant medical context found."

            augmented_prompt = f"""
You are a helpful and knowledgeable AI healthcare assistant.

Use the following retrieved context to answer accurately:

Context:
{context}

User Query:
{query}

Answer the user clearly, factually, and in an empathetic tone.
"""
            return augmented_prompt.strip()
        except Exception as e:
            logger.error(f"Error building augmented prompt: {e}")
            return query

# Wrapper function for backward compatibility
def retrieve_context(query: str, top_k: int = 3) -> str:
    """Return augmented prompt string for given query using RAGPipeline."""
    rag = RAGPipeline()
    return rag.build_augmented_prompt(query)
