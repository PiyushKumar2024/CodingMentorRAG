import os
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

llm = ChatOllama(model="qwen2.5-coder", temperature=0)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are an expert C programming mentor.

Analyze the code strictly.

Only report real issues.

Check ALL of the following:
1. Syntax errors
2. Logical errors
3. Runtime errors (memory, pointers, bounds)
4. Control flow mistakes

Do NOT assume missing context.
Do NOT ignore small syntax issues.
Be precise and concise."""),
    ("user", "Context from Knowledge Base:\n{context}\n\nCode={code}")
])

chain = prompt_template | llm

def generate_answer(code, context):
    try:
        response = chain.invoke({"code": code, "context": context})
        return response.content
    except Exception as e:
        print("Error connecting to LLM:", e)
        return f"Error: {e}"