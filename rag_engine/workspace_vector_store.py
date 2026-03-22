from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

# using nomic's local embeddings here
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

def build_workspace_index(docs, progress_callback=None):
    if not docs:
        raise ValueError("No valid docs to index.")
        
    # start blank with the first doc
    vectorstore = FAISS.from_documents([docs[0]], embeddings)
    if progress_callback:
        progress_callback(1, len(docs))
        
    total = len(docs)
    batch_size = 5 # small batch to keep the UI smooth
    
    # process the rest in chunks so we don't freeze the app
    for i in range(1, total, batch_size):
        batch = docs[i:i+batch_size]
        vectorstore.add_documents(batch)
        if progress_callback:
            progress_callback(min(i+batch_size, total), total)
            
    return vectorstore

def get_workspace_retriever(vectorstore):
    # pull top 5 so deepseek has enough context
    return vectorstore.as_retriever(search_kwargs={"k": 5})
