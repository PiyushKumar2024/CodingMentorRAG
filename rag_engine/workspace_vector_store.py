import os
import hashlib
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

# Using the exact nomic local embedding model specified by the user
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
CACHE_DIR = "vector_cache"

def get_cache_path(workspace_path: str):
    """Generates a unique but deterministic cache folder name for a given workspace path."""
    path_hash = hashlib.md5(workspace_path.encode()).hexdigest()[:8]
    folder_name = os.path.basename(os.path.normpath(workspace_path))
    return os.path.join(CACHE_DIR, f"{folder_name}_{path_hash}")

def build_or_load_workspace_index(workspace_path: str, docs, progress_callback=None):
    """Builds a temporary dynamic FAISS index from codebase chunks, or loads an existing one."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        
    cache_path = get_cache_path(workspace_path)
    
    # Check if we already have it chunked and cached
    if os.path.exists(os.path.join(cache_path, "index.faiss")):
        # Loading local index with dangerous deserialization explicitly allowed because we created it.
        return FAISS.load_local(cache_path, embeddings, allow_dangerous_deserialization=True)
    
    # Build index from scratch with Progress Bar chunking
    if not docs:
        raise ValueError("No valid code documents found to index.")
        
    # Initialize blank FAISS with the first document
    vectorstore = FAISS.from_documents([docs[0]], embeddings)
    if progress_callback:
        progress_callback(1, len(docs))
        
    # Process the rest in batches so Streamlit can update visually
    total = len(docs)
    batch_size = 5 # Small batch size keeps progress bar updating smoothly
    
    for i in range(1, total, batch_size):
        batch = docs[i:i+batch_size]
        vectorstore.add_documents(batch)
        if progress_callback:
            progress_callback(min(i+batch_size, total), total)
            
    # Save the index permanently so it doesn't need to rebuild next time
    vectorstore.save_local(cache_path)
    
    return vectorstore

def get_workspace_retriever(vectorstore):
    """Returns a retriever for the dynamic workspace."""
    # We fetch top 5 related chunks to provide massive context to deepseek-coder
    return vectorstore.as_retriever(search_kwargs={"k": 5})
