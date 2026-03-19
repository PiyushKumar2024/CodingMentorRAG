# 💻 AI Developer Platform (Dual-Pipeline RAG)

A powerful, entirely local AI Developer Platform built to assist software engineers with both targeted, lightning-fast code reviews and massive scale project-level architectural analysis. 

Built with **Streamlit** for a modern, responsive interface and orchestrated by **LangChain** and **Ollama** for entirely private, highly advanced on-device AI. 

---

## 🚀 The Dual-Pipeline Architecture

This project shifts away from the limitations of traditional "dumb" RAG by offering two distinct, highly optimized developer workflows:

### ⚡ Pipeline 1: Quick Code Review (Pure LLM)
Designed for immediate feedback on specific code snippets without the unnecessary overhead of vector searches for basic syntax knowledge.
*   **How it works:** Paste a snippet of C, C++, Python, or Javascript.
*   **DeepSeek Integration:** The backend sends the snippet directly to the `deepseek-coder:6.7b` LLM using an elite Senior Engineer system prompt. 
*   **Output:** Rather than just finding typos, the system performs a deep security audit (OWASP), analyzes Time/Space complexity (Big O), and rewrites the snippet into Best-Practice production code, streaming the response instantly.

### 📂 Pipeline 2: Project Workspace Assistant (Dynamic RAG)
Designed for massive-scale codebase understanding. When working on a full project, the LLM needs context it doesn't natively have. This pipeline dynamically builds that context.
*   **AST-Aware Ingestion:** Upload an absolute path to any project folder. The system uses Langchain's `RecursiveCharacterTextSplitter` configured for code to logically break apart files without splitting classes or functions in half.
*   **Vector Construction (`nomic-embed-text`):** The code chunks are smoothly processed through a live visual loading bar and embedded into a high-dimensional space using local `nomic-embed-text:latest` models.
*   **Persistent FAISS Caching:** If a project has been ingested before, the system skips chunking and instantly loads the local FAISS DB cache file (`vector_cache/`), reducing loading time from minutes to milliseconds.
*   **Contextual Chat:** Ask architectural questions or request new features. The system retrieves the Top 5 structurally relevant chunks from your actual codebase and sends them to DeepSeek-Coder to generate perfectly contextualized answers.

---

## 💻 Installation & Setup Requirements

To run this project, you need **Python** (3.9+) and **Ollama** running locally.

### 1. Install Ollama Models
Ensure Ollama is installed and running (`http://localhost:11434`). You must pull the exact models required by this architecture:
```bash
ollama run deepseek-coder:6.7b
ollama pull nomic-embed-text:latest
```

### 2. Set Up Python Environment
Clone the repository, then create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install LangChain & Streamlit Dependencies
Install the required tools from the heavily optimized `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

With Ollama running in the background, launch the platform:

```bash
streamlit run streamlit_app.py
```

1. Use the **Sidebar** to select your desired workflow.
2. If using **Workspace Assistant**, input the absolute path to your codebase (e.g., `C:/Projects/my-app`) and hit process. 
3. Watch the real-time progress bar build your local vector cache, and then start chatting dynamically with your codebase!

---

## 📂 Project Structure

*   **`streamlit_app.py`**: The main entry point, housing the sidebar routing, the Chat UI, and streaming display logic.
*   **`llm_engine/llm.py`**: Configures the LangChain pipelines, defines the highly-optimized System Prompts, and exposes the streaming generators for DeepSeek.
*   **`rag_engine/`**:
    *   **`project_loader.py`**: Crawler logic and AST-aware semantic text splitters.
    *   **`workspace_vector_store.py`**: Handles building, querying, and persistent disk-caching of the FAISS indices using Nomic embeddings.
