import json
import faiss
import numpy as np
from rag_engine.embed import get_embedding

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

embeddings = [get_embedding(text) for text in texts]

dimension = len(embeddings[0])

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

def search(query, k=2):

    query_embedding = np.array([get_embedding(query)])

    distances, indices = index.search(query_embedding, k)

    results = [texts[i] for i in indices[0]]

    return results