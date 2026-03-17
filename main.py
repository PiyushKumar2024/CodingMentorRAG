from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rag_engine.retrieve import retrieve_context
from llm_engine.llm import generate_answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/mentor")

def coding_mentor(data:dict):

    code = data["code"]

    context = retrieve_context(code)

    answer = generate_answer(code, context)

    return {
        "retrieved_context": context,
        "mentor_response": answer
    }