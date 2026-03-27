import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables 
load_dotenv()
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# Chat LLM — Cerebras Qwen-3 235B
primary_llm = ChatOpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=os.environ.get("CEREBRAS_API_KEY"),
    model="qwen-3-235b-a22b-instruct-2507",
    temperature=0,
    streaming=True
)

# Summarization LLM — Cerebras LLaMA 8B
summary_llm = ChatOpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=os.environ.get("CEREBRAS_API_KEY"),
    model="llama3.1-8b",
    temperature=0,
    streaming=False,
    max_retries=3
)

# Tracks which LLM served the last response
last_provider = "unknown"

summarization_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert developer. Provide a highly detailed 2-3 sentence technical summary of this exact code chunk. 
CRITICAL RULE 1: NEVER use generic boilerplate like 'This is a React component' or 'This file handles data'.
CRITICAL RULE 2: You MUST explicitly write out the precise File Name, exactly what component or main function it exports, what URLs/APIs it hits, and the unique business logic it handles (e.g. 'newcamp.jsx component that renders the form for creating a new Campground entity')."""),
    ("user", "File: {source}\n\nCode:\n{code}")
]) 
summarization_chain = summarization_prompt | summary_llm

def generate_chunk_summary(code, source):
    try:
        response = summarization_chain.invoke({"code": code, "source": source})
        return response.content.strip()
    except Exception:
        return ""

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

<context>
{context}
</context>

CRITICAL RULES:
1. You are strictly forbidden from writing code or explaining concepts that do not exist explicitly inside the <context> block above.
2. If the user asks about a specific file, component, or package (like 'react-hook-form' or 'newcamp') and it is NOT visible in the text inside <context>, you MUST definitively state "I cannot find that in the retrieved context." and stop.
3. NEVER hallucinate code combinations from your pre-training data. 
4. You may provide partial answers ONLY if you are directly quoting from the provided <context>."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{query}")
])

quick_review_chain = quick_review_prompt | primary_llm
workspace_chain = workspace_prompt | primary_llm

def generate_snippet_review_stream(code):
    global last_provider
    try:
        last_provider = "⚡ Cerebras Qwen-3 235B"
        for chunk in quick_review_chain.stream({"code": code}):
            yield chunk.content
    except Exception as e:
        yield f"\n\nError: {e}"

def generate_workspace_answer_stream(query, context, chat_history=None):
    global last_provider
    chat_history = chat_history or []
    formatted_history = [(msg["role"], msg["content"]) for msg in chat_history]
    payload = {"query": query, "context": context, "chat_history": formatted_history}
        
    try:
        last_provider = "⚡ Cerebras Qwen-3 235B"
        for chunk in workspace_chain.stream(payload):
            yield chunk.content
    except Exception as e:
        yield f"\n\nError: {e}"