import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

def load_and_split_project(directory_path):
    suffixes = {".py", ".js", ".jsx", ".ts", ".tsx", ".c", ".cpp", ".h", ".hpp", ".go", ".java", ".rs", ".md", ".html"}
    
    # skip massive build directories
    exclude_dirs = {"node_modules", ".git", "venv", "env", "__pycache__", "dist", "build", "target", "out", ".next"}
    
    docs = []
    
    for root, dirs, files in os.walk(directory_path):
        # modify dirs in-place to completely skip entering excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in suffixes:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if content.strip():
                            docs.append(Document(page_content=content, metadata={"source": file_path}))
                except Exception:
                    # ignore files with weird encodings or read permissions
                    continue

    if not docs:
        return []

    split_docs = []
    for doc in docs:
        ext = os.path.splitext(doc.metadata["source"])[1].lower()
        
        lang = None
        if ext == ".py": lang = Language.PYTHON
        elif ext in [".js", ".jsx"]: lang = Language.JS
        elif ext in [".ts", ".tsx"]: lang = Language.TS
        elif ext in [".cpp", ".hpp"]: lang = Language.CPP
        elif ext in [".c", ".h"]: lang = Language.C
        elif ext == ".go": lang = Language.GO
        elif ext == ".java": lang = Language.JAVA
        elif ext == ".rs": lang = Language.RUST
        elif ext == ".html": lang = Language.HTML
        elif ext == ".md": lang = Language.MARKDOWN

        if lang:
            splitter = RecursiveCharacterTextSplitter.from_language(lang, chunk_size=2500, chunk_overlap=500)
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)

        split_docs.extend(splitter.split_documents([doc]))

    return split_docs
