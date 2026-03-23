import streamlit as st
import os

from llm_engine.llm import generate_snippet_review_stream, generate_workspace_answer_stream
from rag_engine.project_loader import load_and_split_project
from rag_engine.workspace_vector_store import build_workspace_index, get_workspace_retriever

st.set_page_config(page_title="Developer Platform", page_icon="💻", layout="wide")

with st.sidebar:
    st.title("Navigation")
    mode = st.radio("Workflow:", ["Quick Code Review", "Workspace Assistant"])
    st.markdown("---")

if mode == "Quick Code Review":
    st.title("Quick Code Review")
    st.markdown("Paste code for immediate LLM review.")
    
    code_input = st.text_area("Code Input", height=250)
    
    if st.button("Run Review", type="primary"):
        if not code_input.strip():
            st.warning("Please enter some code first.")
        else:
            st.subheader("Review")
            # stream response directly to UI
            st.write_stream(generate_snippet_review_stream(code_input))

elif mode == "Workspace Assistant":
    st.title("Workspace Assistant")
    st.markdown("Chat with your codebase using RAG.")
    
    workspace_path = st.text_input("Absolute path to project folder")
    
    if st.button("Process Workspace"):
        if not workspace_path or not os.path.exists(workspace_path):
            st.error("Please provide a valid directory path.")
        else:
            with st.spinner("Loading files..."):
                docs = load_and_split_project(workspace_path)
                
            if docs:
                progress_bar = st.progress(0, text="Indexing...")
                
                # small callback to push updates to progress bar
                def update_progress(current, total):
                    percent = min(current / total, 1.0)
                    progress_bar.progress(percent, text=f"Indexing ({current}/{total})")
                    
                vectorstore = build_workspace_index(docs, progress_callback=update_progress)
                progress_bar.empty() # clear it when done
                
                st.session_state["workspace_retriever"] = get_workspace_retriever(vectorstore)
                st.success(f"Processed: {workspace_path}")
                st.session_state["workspace_ready"] = True
            else:
                st.error("No valid code files found.")
    
    if st.session_state.get("workspace_ready"):
        st.markdown("---")
        st.subheader("Chat")
        
        # init chat history
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
            
        # render past messages
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        query = st.chat_input("Ask a question...")
        if query:
            with st.chat_message("user"):
                st.markdown(query)
            st.session_state["messages"].append({"role": "user", "content": query})
            
            with st.chat_message("assistant"):
                retriever = st.session_state["workspace_retriever"]
                retrieved_docs = retriever.invoke(query)
                
                # Debug UI: show exactly what files FAISS retrieved before answering
                with st.expander(f"🔍 System analyzed {len(retrieved_docs)} retrieved code chunks for this question"):
                    for doc in retrieved_docs:
                        st.caption(f"- `{doc.metadata.get('source', 'Unknown file')}`")
                
                # stitch together the context chunks
                context = ""
                for doc in retrieved_docs:
                    source_file = doc.metadata.get("source", "Unknown file")
                    context += f"\n--- {source_file} ---\n{doc.page_content}\n"
                
                # strip the latest message from history to prevent duplication
                history_for_llm = st.session_state["messages"][:-1]
                answer = st.write_stream(generate_workspace_answer_stream(query, context, history_for_llm))
                
            st.session_state["messages"].append({"role": "assistant", "content": answer})
