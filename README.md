# 💻 AI Developer Platform (Hybrid RAG Architecture)

A powerful, hybrid-architecture AI Developer Platform built to assist software engineers with both targeted, lightning-fast code reviews and massive scale project-level architectural analysis. 

Built with **Streamlit** for a modern, responsive interface and orchestrated by **LangChain**, **Ollama**, and **Cerebras Cloud** to strike the perfect balance between local privacy, zero-cost API rate survivability, and massive 70-Billion-Parameter intellect.

---

## 🚀 The Hybrid Pipeline Architecture

This project explicitly solves the "Hallucination" and "Context Starvation" problems found in traditional RAG by utilizing an advanced multi-tiered approach:

### ⚡ Pipeline 1: Quick Code Review 
Designed for immediate feedback on specific code snippets without the unnecessary overhead of vector searches.
*   **DeepSeek Integration:** The backend sends the snippet directly to a specialized LLM using an elite Senior Engineer system prompt. 
*   **Output:** Rather than just finding typos, the system performs a deep security audit (OWASP), analyzes Time/Space complexity (Big O), and rewrites the snippet into Best-Practice production code, streaming the response instantly.

### 📂 Pipeline 2: Project Workspace Assistant (Zero-Hallucination RAG)
Designed for massive-scale codebase understanding and context retrieval.
*   **AST-Aware Ingestion:** Upload an absolute path to any project folder. The system uses Langchain's `RecursiveCharacterTextSplitter` configured for code to logically break apart massive files.
*   **Summary-Augmented Generation (SAG):** Before hitting the database, every chunk is sent to an incredibly fast 8B model to generate a strict, keyword-dense functional summary. 
*   **Hardcoded Vector Paths:** The absolute file path of the chunk is computationally fused directly into the FAISS mathematical embedding block. This explicitly solves mathematical "snowblindness" and allows the engine to instantly locate exact files based on specific prompt phrasing.
*   **Transparent UI Debugging:** The system natively renders a Streamlit expander allowing you to see precisely which 8 code chunks FAISS grabbed from your filesystem.
*   **LLaMA-70B Chat Engine:** When chatting, your query and the retrieved context are instantly forwarded to the bleeding-edge **Cerebras LLaMA-3.1-70B** model. Strictly controlled via XML constraints, the 70B model analyzes your entire architecture and refuses to hallucinate missing imports.

---

## 💻 Installation & Setup Requirements

To run this project, you need **Python (3.9+)**, **Ollama**, and a free **Cerebras API Key**.

### 1. Install Ollama Models (Local Fallback & Embeddings)
Ensure Ollama is installed and running (`http://localhost:11434`). You must pull the local vector embedding model:
```bash
ollama pull nomic-embed-text:latest
ollama run deepseek-coder:6.7b
```

### 2. Set Up API Keys
The system relies on Cerebras for instant, free-tier LLaMA 70B inference. Get an API key from `cloud.cerebras.ai`. Provide it inside `llm_engine/llm.py` or export it into your runtime environment:
```bash
export CEREBRAS_API_KEY="your-key-here"
```

### 3. Set Up Python Environment
Clone the repository, then create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

With Ollama running in the background:

```bash
streamlit run streamlit_app.py
```

1. Use the **Sidebar** to select your desired workflow.
2. If using **Workspace Assistant**, input the absolute path to your codebase (e.g., `C:/Projects/my-app`) and click "Process Workspace". 
3. Watch the real-time progress bar build your local vector cache, and pop open the `workspace_index_debug.txt` file generated in your directory to read the AI's internal thoughts as it maps your project!

---

## 📂 Project Structure

*   **`streamlit_app.py`**: The main entry point, housing the UI routing, debug expanders, and chat state logic.
*   **`llm_engine/llm.py`**: Configures the LangChain pipelines, XML anti-hallucination guardrails, and the Multi-Model (70B vs 8B) routing architecture.
*   **`rag_engine/`**:
    *   **`project_loader.py`**: Crawler logic and AST-aware semantic text splitters.
    *   **`workspace_vector_store.py`**: Handles building, querying, and persistent disk-streaming of the FAISS indices using Nomic embeddings and generated summaries.
