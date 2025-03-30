#!/usr/bin/env python3
"""Script to update the RAG processing code to work with the new environment."""
import os
import re

# Path to the processing.py file that needs updating
processing_file = "rag/processing.py"

def update_textract_imports():
    """Update the textract imports to make them optional."""
    with open(processing_file, 'r') as f:
        content = f.read()
    
    # Make textract an optional import
    modified = re.sub(
        r'import magic, textract, fitz, pytesseract',
        'import magic, fitz, pytesseract\n'
        'try:\n'
        '    import textract\n'
        '    HAS_TEXTRACT = True\n'
        'except ImportError:\n'
        '    HAS_TEXTRACT = False\n'
        '    print("Warning: textract module not available. Some document types may not be processed correctly.")',
        content
    )
    
    # Update the file processing methods to check for textract
    modified = re.sub(
        r'return textract\.process\(str\(path\), \*\*self\.textract_config\)\.decode\(\'utf-8\'\)',
        'if HAS_TEXTRACT:\n'
        '            return textract.process(str(path), **self.textract_config).decode(\'utf-8\')\n'
        '        else:\n'
        '            print(f"Cannot process {path}: textract not available")\n'
        '            return None',
        modified
    )
    
    # Also update the generic processing method
    modified = re.sub(
        r'return textract\.process\(str\(path\)\)\.decode\(\'utf-8\', errors=\'ignore\'\)',
        'if HAS_TEXTRACT:\n'
        '                return textract.process(str(path)).decode(\'utf-8\', errors=\'ignore\')\n'
        '            else:\n'
        '                print(f"Cannot process {path}: textract not available")\n'
        '                return None',
        modified
    )
    
    # Update pinecone imports
    modified = re.sub(
        r'import pinecone',
        'import pinecone  # Make sure to use "pinecone" not "pinecone-client"',
        modified
    )
    
    # Write the modified content back to the file
    with open(processing_file, 'w') as f:
        f.write(modified)
    
    print(f"Updated {processing_file} with optional textract imports")

if __name__ == "__main__":
    update_textract_imports()
