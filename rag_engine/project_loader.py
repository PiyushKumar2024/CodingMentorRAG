import os
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser

def load_and_split_project(directory_path: str):
    # just grab the most common extensions
    suffixes = [".py", ".js", ".jsx", ".ts", ".tsx", ".c", ".cpp", ".h", ".hpp", ".go", ".java", ".rs", ".md", ".html"]
    
    # generic loader doesn't choke on binary stuff by default
    loader = GenericLoader.from_filesystem(
        directory_path,
        glob="**/*",
        suffixes=suffixes,
        parser=LanguageParser()
    )
    docs = loader.load()
    if not docs:
        return []

    split_docs = []
    for doc in docs:
        ext = os.path.splitext(doc.metadata.get("source", ""))[1].lower()
        
        # map extension to lang enum so the splitter knows what to do
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

        # fallback to regular chunking if we don't know the language AST
        if lang:
            splitter = RecursiveCharacterTextSplitter.from_language(lang, chunk_size=1000, chunk_overlap=200)
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        split_docs.extend(splitter.split_documents([doc]))

    return split_docs
