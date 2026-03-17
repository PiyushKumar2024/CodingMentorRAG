import json
import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_vector_store():
    with open("knowledge_base/coding_kb.json") as f:
        docs = json.load(f)

    texts = [
        f"""
    Topic: {d['topic']}
    Description: {d['description']}
    Bad Example: {d['bad_example']}
    Fix: {d['fix']}
    Explanation: {d['explanation']}
    """
        for d in docs
    ]
    
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore

vectorstore = load_vector_store()

def search(query, k=2):
    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]