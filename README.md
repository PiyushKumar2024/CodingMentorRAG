# AI Coding Mentor (RAG-Powered)

An intelligent, context-aware AI Coding Mentor designed to analyze C programming code and provide expert-level feedback. This project integrates Retrieval-Augmented Generation (RAG) to fetch relevant programming knowledge and feeds it to a locally hosted Large Language Model (LLM) to deliver highly precise, actionable code reviews.

Built with **Streamlit** for an interactive modern frontend and orchestrated by **LangChain** for robust LLM interactions.

## 🚀 Key Features

*   **Interactive Web Interface:** Paste your C code and instantly receive analysis directly in the browser via Streamlit.
*   **Deep Code Analysis:** Identifies syntax errors, logical flaws, runtime issues (memory, pointers, bounds), and control flow mistakes.
*   **Retrieval-Augmented Generation (RAG):**
    *   Uses **Sentence Transformers** (`all-MiniLM-L6-v2`) to convert code into vector embeddings.
    *   Leverages **FAISS** to perform rapid similarity searches against a local knowledge base (`knowledge_base/coding_kb.json`) containing common programming pitfalls and best practices.
*   **Local LLM Integration:** Powered by the `qwen2.5-coder` model running locally via **Ollama**, ensuring privacy, zero API costs, and low latency.
*   **LangChain Orchestration:** Clean, scalable architecture for constructing prompts and managing the LLM pipeline.

## 🛠️ Architecture Overview

1.  **User Input:** The user submits a snippet of C code via the Streamlit interface.
2.  **Context Retrieval (RAG):** The input code is embedded into a vector. FAISS searches the `coding_kb.json` base to pull the most relevant constraints, rules, or historical common errors matching the code context.
3.  **LLM Generation:** LangChain combines the raw user code and the retrieved RAG context into a strict expert-mentor prompt.
4.  **Local Inference:** The prompt is sent to `qwen2.5-coder` running inside Ollama.
5.  **Mentor Output:** The precise, concise review and suggested fixes are streamed back to the user interface.

## 💻 Installation & Setup Requirements

To run this project locally, you will need Python 3.9+ and Ollama installed on your system.

### 1. Install Ollama and the LLM Model
Ollama is required to run the `qwen2.5-coder` language model locally.
1.  Download and install Ollama from [ollama.ai](https://ollama.ai/).
2.  Open your terminal or command prompt and pull the specific coding model required for this project:
    ```bash
    ollama run qwen2.5-coder
    ```
    *Keep Ollama running in the background. It will expose a local server at `http://localhost:11434`.*

### 2. Clone the Repository
Open a new terminal and clone this repository:
```bash
git clone <YOUR_NEW_REPOSITORY_URL_HERE>
cd CodingMentorRAG
```

### 3. Set Up Python Environment (Recommended)
It's highly recommended to use a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
Install all required Python packages (including Streamlit, LangChain, FAISS, and Sentence Transformers):
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

With Ollama running in the background and your virtual environment activated, start the Streamlit application:

```bash
streamlit run streamlit_app.py
```

This will automatically open a new tab in your default web browser (usually at `http://localhost:8501`).

### How to Use:
1. Paste your C code (e.g., a loop, memory allocation, or array manipulation) into the provided text area.
2. Click **"Analyze Code"**.
3. The AI mentor will process the request, retrieve relevant knowledge from the FAISS database, query the Ollama model, and stream the analysis back to you.

## 📂 Project Structure

*   **`streamlit_app.py`**: The main entry point for the Streamlit web application.
*   **`llm_engine/llm.py`**: Contains the LangChain setup and LLM invocation logic.
*   **`rag_engine/`**:
    *   `vector_store.py`: Initializes the FAISS index and handles the logic for storing and querying the knowledge base.
    *   `retrieve.py`: A utility module to fetch and format the top relevant documents.
*   **`knowledge_base/coding_kb.json`**: A JSON file acting as the knowledge base, documenting various coding rules, bad examples, and explanations.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue if you encounter bugs or have suggestions for improvements.
