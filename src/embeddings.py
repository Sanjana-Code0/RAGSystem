# Optimized version
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os

INDEX_DIR = "faiss_index"

def get_embeddings_model():
    """Return the specified HuggingFace embeddings model."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )

def build_vectorstore(chunks: list[Document]) -> FAISS:
    """Creates FAISS index from document chunks and saves it."""
    embeddings = get_embeddings_model()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save the index locally
    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)
    
    return vectorstore

def load_vectorstore() -> FAISS:
    """Loads FAISS index from the local directory."""
    if not os.path.exists(INDEX_DIR):
        raise FileNotFoundError(f"FAISS index folder '{INDEX_DIR}' not found. Please build the index first.")
        
    embeddings = get_embeddings_model()
    # allow_dangerous_deserialization is needed in newer LangChain versions for FAISS
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return vectorstore
