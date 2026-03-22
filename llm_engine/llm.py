import os
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# setup model w/ streaming on for immediate feedback
llm = ChatOllama(model="deepseek-coder:6.7b", temperature=0, streaming=True)

quick_review_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior developer. Review the code snippet below.
Provide a professional analysis covering:
1. Security vulnerabilities
2. Time/Space Complexity
3. Code Quality 
4. A refactored version of the code."""),
    ("user", "{code}")
])

workspace_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant helping with a project workspace.
Answer questions or write code based on the provided context.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{query}")
])

quick_review_chain = quick_review_prompt | llm
workspace_chain = workspace_prompt | llm

def generate_snippet_review_stream(code: str):
    try:
        for chunk in quick_review_chain.stream({"code": code}):
            yield chunk.content
    except Exception as e:
        yield f"\n\nError: {e}"

def generate_workspace_answer_stream(query: str, context: str, chat_history: list = None):
    chat_history = chat_history or []
    # format dicts into tuples for langchain
    formatted_history = [(msg["role"], msg["content"]) for msg in chat_history]
        
    try:
        for chunk in workspace_chain.stream({
            "query": query, 
            "context": context, 
            "chat_history": formatted_history
        }):
            yield chunk.content
    except Exception as e:
        yield f"\n\nError: {e}"