from rag_engine.vector_store import search

def retrieve_context(code):

    docs = search(code,2)

    context = "\n".join(docs)

    return context