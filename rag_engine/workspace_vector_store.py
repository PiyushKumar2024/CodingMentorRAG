from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from llm_engine.llm import generate_chunk_summary
import time

# using nomic's local embeddings here
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

def build_workspace_index(docs, progress_callback=None):
    if not docs:
        raise ValueError("No valid docs to index.")
        
    total = len(docs)
    
    # NEW: Create a brand new debug file for the user to visibly inspect what AI is writing
    debug_path = "workspace_index_debug.txt"
    open(debug_path, "w", encoding="utf-8").close()
    
    # strictly sleep 2.1s to guarantee we never exceed Cerebras 30 Request/Minute free tier
    time.sleep(2.1)
    
    # summarize the first doc before starting blank vectorstore
    first_summary = generate_chunk_summary(docs[0].page_content, docs[0].metadata.get("source", "Unknown"))
    if first_summary:
        file_src = docs[0].metadata.get('source', 'Unknown')
        docs[0].page_content = f"FILE PATH: {file_src}\nSUMMARY: {first_summary}\n\nCODE:\n{docs[0].page_content}"
        
    # Write first parsed doc to the debug file
    with open(debug_path, "a", encoding="utf-8") as debug_file:
        debug_file.write(f"--- FILE: {docs[0].metadata.get('source')} ---\n{docs[0].page_content}\n=========================================\n\n")
        
    vectorstore = FAISS.from_documents([docs[0]], embeddings)
    if progress_callback:
        progress_callback(1, total)
        
    batch_size = 5 # small batch to keep the UI smooth
    
    # process the rest in chunks so we don't freeze the app
    for i in range(1, total, batch_size):
        batch = docs[i:i+batch_size]
        
        for doc in batch:
            time.sleep(2.1) 
            summary = generate_chunk_summary(doc.page_content, doc.metadata.get("source", "Unknown"))
            if summary:
                file_src = doc.metadata.get('source', 'Unknown')
                doc.page_content = f"FILE PATH: {file_src}\nSUMMARY: {summary}\n\nCODE:\n{doc.page_content}"
                
        # Bulk append the chunks to read the index in VS Code
        with open(debug_path, "a", encoding="utf-8") as debug_file:
            for doc in batch:
                debug_file.write(f"--- FILE: {doc.metadata.get('source')} ---\n{doc.page_content}\n=========================================\n\n")
                
        vectorstore.add_documents(batch)
        if progress_callback:
            progress_callback(min(i+batch_size, total), total)
            
    return vectorstore

def get_workspace_retriever(vectorstore):
    # k=8 safely grabs tightly coupled files without exceeding the strict 8192 context window of LLaMA 8B
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})
