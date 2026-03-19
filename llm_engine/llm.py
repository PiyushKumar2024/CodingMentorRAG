import os
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# Using deepseek-coder:6.7b with streaming enabled
llm = ChatOllama(model="deepseek-coder:6.7b", temperature=0, streaming=True)

# 1. Pure LLM Prompt (No RAG) for Quick Reviews
quick_review_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an elite Senior Staff Software Engineer and Security Auditor.
Your job is to review the provided code snippet.
Do NOT just point out basic syntax errors—provide a deep, professional analysis.

Your review MUST cover:
1. Security vulnerabilities (OWASP, buffer overflows, injection, etc.)
2. Time and Space Complexity (Big O)
3. Code Quality & Clean Code principles (SOLID, DRY)
4. A complete, refactored Best-Practice version of the code.

Be blunt, precise, and extremely thorough."""),
    ("user", "Code to review:\n{code}")
])

# 2. Workspace Assistant Prompt (With Dynamic RAG)
workspace_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert AI Developer perfectly integrated into the user's project workspace.
Use the provided codebase context to answer the user's architectural questions or generate new files/features that perfectly match the existing project patterns.

Codebase Context:
{context}

If asked to generate code, provide completely working replacements or new files. Do NOT make up files that don't exist."""),
    ("user", "User Request:\n{query}")
])

quick_review_chain = quick_review_prompt | llm
workspace_chain = workspace_prompt | llm

def generate_snippet_review_stream(code: str):
    try:
        for chunk in quick_review_chain.stream({"code": code}):
            yield chunk.content
    except Exception as e:
        yield f"\n\nError connecting to LLM: {e}"

def generate_workspace_answer_stream(query: str, context: str):
    try:
        for chunk in workspace_chain.stream({"query": query, "context": context}):
            yield chunk.content
    except Exception as e:
        yield f"\n\nError connecting to LLM: {e}"