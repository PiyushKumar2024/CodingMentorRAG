import streamlit as st
import os

from llm_engine.llm import generate_snippet_review_stream, generate_workspace_answer_stream
from rag_engine.project_loader import load_and_split_project
from rag_engine.workspace_vector_store import build_workspace_index, get_workspace_retriever

st.set_page_config(page_title="AI Developer Platform", page_icon="💻", layout="wide")

# Sidebar for Mode Selection
with st.sidebar:
    st.title("Navigation")
    mode = st.radio(
        "Choose your workflow:",
        ["⚡ Quick Code Review", "📂 Workspace Assistant"]
    )
    st.markdown("---")
    st.markdown("**Powered by DeepSeek-Coder & Dynamic RAG**")

if mode == "⚡ Quick Code Review":
    st.title("⚡ Quick Code Review")
    st.markdown("Paste a snippet for an immediate, Pure LLM deep-dive (Security, Big O, Refactoring). No RAG overhead.")
    
    code_input = st.text_area("C/C++/Python/JS Code Input", height=250, placeholder="Paste your code here...")
    
    if st.button("Run Deep Review", type="primary"):
        if not code_input.strip():
            st.warning("Please enter some code first.")
        else:
            st.subheader("Senior Engineer Analysis")
            # Uses write_stream to display text token-by-token instantly
            st.write_stream(generate_snippet_review_stream(code_input))

elif mode == "📂 Workspace Assistant":
    st.title("📂 Project Workspace Assistant")
    st.markdown("Upload your project folder to chat with your codebase using Dynamic Project-Level RAG.")
    
    workspace_path = st.text_input("Absolute path to your project folder (e.g., C:/Projects/my-app)")
    
    if st.button("Process Workspace"):
        if not workspace_path or not os.path.exists(workspace_path):
            st.error("Please provide a valid, existing directory path.")
        else:
            with st.spinner("Ingesting codebase files from directory..."):
                docs = load_and_split_project(workspace_path)
                
            if docs:
                # Create a Streamlit progress bar
                progress_bar = st.progress(0, text="Embedding and Indexing Code Chunks...")
                
                def update_progress(current, total):
                    percent = min(current / total, 1.0)
                    progress_bar.progress(percent, text=f"Embedding Chunks... ({current}/{total})")
                    
                # Build index while calling update_progress iteratively
                vectorstore = build_workspace_index(docs, progress_callback=update_progress)
                
                progress_bar.empty() # Remove the UI bar when finished
                
                st.session_state["workspace_retriever"] = get_workspace_retriever(vectorstore)
                st.success(f"Successfully processed workspace: {workspace_path} ({len(docs)} chunks embedded & saved)")
                st.session_state["workspace_ready"] = True
            else:
                st.error("No valid code files (.c, .py, .js, .cpp) found in the directory.")
    
    if st.session_state.get("workspace_ready"):
        st.markdown("---")
        st.subheader("Chat with your Codebase")
        
        # Initialize chat history state
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
            
        # Display chat messages from history on app rerun
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        query = st.chat_input("Ask about architecture, or tell me to write a new feature...")
        if query:
            # Render user query and save it immediately
            with st.chat_message("user"):
                st.markdown(query)
            st.session_state["messages"].append({"role": "user", "content": query})
            
            with st.chat_message("assistant"):
                # Retrieve files from our dynamic FAISS index
                retriever = st.session_state["workspace_retriever"]
                retrieved_docs = retriever.invoke(query)
                
                # Construct formatted context string indicating source files
                context = ""
                for doc in retrieved_docs:
                    source_file = doc.metadata.get("source", "Unknown file")
                    context += f"\n--- Source: {source_file} ---\n{doc.page_content}\n"
                
                # Pass history (excluding the very last item which is the current user query)
                history_for_llm = st.session_state["messages"][:-1]
                
                # Stream the response utilizing the contextual pipeline
                answer = st.write_stream(generate_workspace_answer_stream(query, context, history_for_llm))
                
            # Append AI response to track the memory
            st.session_state["messages"].append({"role": "assistant", "content": answer})
