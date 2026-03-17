import streamlit as st
from rag_engine.retrieve import retrieve_context
from llm_engine.llm import generate_answer

st.set_page_config(page_title="AI Coding Mentor", page_icon="👨‍💻", layout="wide")

st.title("👨‍💻 AI Coding Mentor")
st.markdown("Submit your C code below for an expert, AI-powered review using LangChain and Streamlit capabilities.")

code_input = st.text_area("C Code Input", height=250, placeholder="Paste your C code here...")

if st.button("Analyze Code"):
    if not code_input.strip():
        st.warning("Please enter some code first before analyzing.")
    else:
        with st.spinner("Analyzing your code with the RAG engine..."):
            context = retrieve_context(code_input)
            
            st.success("Context retrieved from the knowledge base successfully!")
            
            with st.spinner("Generating expert feedback..."):
                answer = generate_answer(code_input, context)
            
            st.markdown("---")
            st.subheader("Analysis Result")
            st.markdown(answer)
            
            with st.expander("View Retrieved Context (RAG)"):
                st.text(context)
