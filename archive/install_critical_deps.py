#!/usr/bin/env python3
"""Script to install critical dependencies for the RAG pipeline."""
import os
import subprocess
import sys
import time

def run_command(command):
    """Run a command and print its output in real-time."""
    print(f"Running: {command}")
    with subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, universal_newlines=True
    ) as process:
        for line in process.stdout:
            print(line, end='')
        process.wait()
        return process.returncode

def main():
    # Get the virtual environment python executable path
    venv_python = sys.executable
    
    # Core packages needed for the RAG pipeline
    core_packages = [
        "openai",
        "pinecone-client",
        "langchain",
        "langchain-core",
        "transformers",
        "torch",
        "PyMuPDF",
        "tiktoken",
        "python-dotenv",
        "python-magic",
        "pytesseract",
        "numpy",
        "pandas",
        "pillow"
    ]
    
    # Install core packages
    print("Installing core packages...")
    for package in core_packages:
        print(f"\nInstalling {package}...")
        result = run_command(f"{venv_python} -m pip install {package} --no-cache-dir")
        if result != 0:
            print(f"Failed to install {package}! Error code: {result}")
            continue
        
    print("\nInstallation complete!")
    print("\nNOTE: 'textract' was not installed due to dependency conflicts.")
    print("Your RAG pipeline should work for PDF and other basic document types.")
    print("If you need more document type support, you may need to manually install specific libraries.")

if __name__ == "__main__":
    main()
