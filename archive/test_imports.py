#!/usr/bin/env python
"""
Test script to check if key imports are working
"""
import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Test basic imports
modules_to_test = [
    "pinecone",
    "openai",
    "torch",
    "transformers",
    "langchain",
    "textract",
    "magic",
    "fitz",
    "pytesseract"
]

for module in modules_to_test:
    try:
        print(f"Trying to import {module}...", end="")
        imported_module = __import__(module)
        print(" SUCCESS")
        
        # Special verification for textract
        if module == "textract":
            print("  Testing textract functionality...")
            try:
                # Create a simple test text file
                with open("textract_test.txt", "w") as f:
                    f.write("This is a test document for textract.")
                
                # Try to extract text
                extracted_text = imported_module.process("textract_test.txt").decode('utf-8')
                print(f"  Textract extraction: {extracted_text.strip()}")
                
                # Clean up test file
                import os
                os.remove("textract_test.txt")
            except Exception as e:
                print(f"  Textract test failed: {str(e)}")
                
    except ImportError as e:
        print(f" FAILED: {str(e)}")
    except Exception as e:
        print(f" ERROR: {str(e)}")

# Check if pip is working properly
print("\nChecking if pip internals are working...")
try:
    from pip._internal.cli import main as pip_main
    print("Pip internals can be imported")
except ImportError as e:
    print(f"Pip import failed: {str(e)}")
