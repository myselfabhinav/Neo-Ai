import os
import pickle
import numpy as np
from typing import List, Tuple
import faiss
import logging

# -------------------- Setup -------------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

# Dummy EmbeddingModel for type hints
class EmbeddingModel:
    """Wrapper for embedding models used in VectorStore."""
    def get_embeddings(self, texts: List[str]):
        return np.random.rand(len(texts), 384)  # Example: 384-dimensional embeddings

    def get_embedding_dim(self):
        return 384

# -------------------- VectorStore -------------------- #
class VectorStore:
    """FAISS-based vector store for storing and retrieving document embeddings."""
    def __init__(self, embedding_model: EmbeddingModel, index_path: str = "data/faiss_index.bin", metadata_path: str = "data/metadata.pkl"):
        self.embedding_model = embedding_model
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []

        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.load_index()
        else:
            self.create_new_index()

    def create_new_index(self):
        embedding_dim = self.embedding_model.get_embedding_dim()
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.metadata = []

    def add_documents(self, docs: List[str]):
        embeddings = np.array(self.embedding_model.get_embeddings(docs)).astype("float32")
        self.index.add(embeddings)
        self.metadata.extend(docs)
        self.save_index()

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        query_vector = np.array(self.embedding_model.get_embeddings([query])).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                similarity = 1 / (1 + dist)
                results.append((self.metadata[idx], float(similarity)))
        return results

    def save_index(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load_index(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    def clear_index(self):
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        self.create_new_index()

# -------------------- Helper Function -------------------- #
def load_or_create_vectorstore(embedding_model: EmbeddingModel, docs_path: str, vectorstore_path: str):
    try:
        vs = VectorStore(embedding_model, index_path=vectorstore_path, metadata_path=vectorstore_path + ".meta")
        if not vs.metadata and os.path.exists(docs_path):
            from glob import glob
            files = glob(os.path.join(docs_path, "*.txt"))
            docs = [open(f, "r", encoding="utf-8").read() for f in files]
            if docs:
                vs.add_documents(docs)
        logging.info("✅ Vectorstore loaded or created successfully.")
        return vs
    except Exception as e:
        logging.error(f"❌ Failed to load/create vectorstore: {str(e)}")
        return None

# -------------------- Test -------------------- #
if __name__ == "__main__":
    em = EmbeddingModel()
    vs = load_or_create_vectorstore(em, docs_path="data/docs", vectorstore_path="data/vectorstore.bin")
    print(vs)
