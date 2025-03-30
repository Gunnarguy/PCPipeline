#!/usr/bin/env python3
"""
Test script to verify that the core RAG functionality works.
This script only initializes the key components without running the full application.
"""
import os
import sys
from pathlib import Path

def main():
    """Initialize the core components of the RAG system to verify they work."""
    print("Testing RAG components initialization...")
    
    # Initialize OpenAI
    try:
        print("\nTesting OpenAI initialization...")
        from openai import OpenAI
        # Just create the client, don't make actual API calls
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "sk-dummy-key-for-testing"))
        print("✅ OpenAI client initialized successfully")
    except Exception as e:
        print(f"❌ OpenAI initialization failed: {e}")
    
    # Initialize Pinecone
    try:
        print("\nTesting Pinecone initialization...")
        import pinecone
        # Just create the client, don't make actual API calls (using correct package)
        pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY", "dummy-key-for-testing"))
        print("✅ Pinecone client initialized successfully")
    except Exception as e:
        print(f"❌ Pinecone initialization failed: {e}")
    
    # Test PDF processing components
    try:
        print("\nTesting PDF processing components...")
        import fitz  # PyMuPDF
        print("✅ PyMuPDF (fitz) imported successfully")
    except Exception as e:
        print(f"❌ PyMuPDF import failed: {e}")
    
    # Test langchain components
    try:
        print("\nTesting LangChain components...")
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        print("✅ LangChain text splitter initialized successfully")
    except Exception as e:
        print(f"❌ LangChain components failed: {e}")
    
    # Test transformers/reranker components
    try:
        print("\nTesting transformers components...")
        import torch
        import transformers
        print(f"✅ Transformers v{transformers.__version__} imported successfully")
        print(f"✅ PyTorch v{torch.__version__} imported successfully")
    except Exception as e:
        print(f"❌ Transformers/PyTorch import failed: {e}")
    
    # Test essential utility packages
    try:
        print("\nTesting utility packages...")
        import numpy
        import pandas
        print(f"✅ NumPy v{numpy.__version__} imported successfully")
        print(f"✅ Pandas v{pandas.__version__} imported successfully")
    except Exception as e:
        print(f"❌ Utility packages import failed: {e}")
    
    print("\nVerification complete!")

if __name__ == "__main__":
    main()
