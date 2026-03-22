from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

# Using the exact nomic local embedding model specified by the user
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

def build_workspace_index(docs, progress_callback=None):
    """Builds a temporary dynamic FAISS index from codebase chunks."""
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
            
    return vectorstore

def get_workspace_retriever(vectorstore):
    """Returns a retriever for the dynamic workspace."""
    # We fetch top 5 related chunks to provide massive context to deepseek-coder
    return vectorstore.as_retriever(search_kwargs={"k": 5})
