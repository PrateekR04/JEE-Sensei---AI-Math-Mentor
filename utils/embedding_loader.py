"""
Embedding Loader
Handles persistent local caching and loading of the embedding model.
Ensures the model is downloaded only once and loaded from disk thereafter.
"""

import os
import sys
from pathlib import Path
import streamlit as st

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

# Configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_BASE_DIR = Path("models/embeddings")
LOCAL_MODEL_PATH = CACHE_BASE_DIR / "all-MiniLM-L6-v2"

@st.cache_resource
def load_embedding_model():
    """
    Load the embedding model.
    - Checks if local cache exists at ./models/embeddings/all-MiniLM-L6-v2
    - If yes, loads from disk
    - If no, downloads from HuggingFace, saves to disk, then loads
    
    Returns:
        SentenceTransformer model instance or None if library missing
    """
    if not HAS_SENTENCE_TRANSFORMERS:
        print("Warning: sentence-transformers not installed.")
        return None
        
    # Ensure cache directory exists
    CACHE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if model exists locally
    if not LOCAL_MODEL_PATH.exists():
        print(f"📥 Downloading embedding model and caching to {LOCAL_MODEL_PATH}...")
        try:
            # Download and save
            model = SentenceTransformer(MODEL_NAME)
            model.save(str(LOCAL_MODEL_PATH))
            print("✅ Embedding model downloaded and cached successfully.")
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            raise e
    else:
        print("📦 Embedding model loaded from local cache.")
    
    # Load from local path
    try:
        # Load from the local directory
        model = SentenceTransformer(str(LOCAL_MODEL_PATH), device="cpu")
        return model
    except Exception as e:
        print(f"❌ Error loading cached model: {e}")
        # Fallback: try to redownload if cache is corrupted
        try:
            print("⚠️ Cache might be corrupted, re-downloading...")
            model = SentenceTransformer(MODEL_NAME)
            model.save(str(LOCAL_MODEL_PATH))
            return model
        except Exception as retry_e:
            raise retry_e

def get_local_model_path() -> str:
    """
    Ensure the model is cached and return the absolute local path.
    Useful for libraries that expect a path string (e.g., LangChain).
    
    Returns:
        Absolute path to the local model directory
    """
    # Trigger loading to ensure it exists
    _ = load_embedding_model()
    return str(LOCAL_MODEL_PATH.absolute())
