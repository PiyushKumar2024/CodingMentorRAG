import os
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser

def load_and_split_project(directory_path: str):
    """
    Loads all C, C++, Python, and JS files from a directory 
    and splits them logically using AST/Syntax-aware splitters.
    """
    
    # Generic loader that ignores binary files and hidden folders by default
    loader = GenericLoader.from_filesystem(
        directory_path,
        glob="**/*",
        suffixes=[".c", ".cpp", ".py", ".js"],
        parser=LanguageParser()
    )
    
    docs = loader.load()
    
    if not docs:
        return []

    # Use a generic code splitter (Python setup works well universally for bracketed/indented languages)
    # This prevents blindly cutting a function in half by respecting '\nclass ', '\ndef ', etc.
    code_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, 
        chunk_size=1000, 
        chunk_overlap=200
    )
    
    split_docs = code_splitter.split_documents(docs)
    return split_docs
