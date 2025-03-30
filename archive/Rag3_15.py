#!/usr/bin/env python3
"""
Refactored version of ultimate_rag_final.py (Version: 6.3 - Chunking Enhanced)
Improvements over 6.2:
- Implemented adaptive chunking strategies based on MIME type
- Added structured PDF processing to preserve document hierarchy (headings, etc.)
- Included placeholder functions for semantic and query-aware chunking (for future enhancement)
- Enhanced comments and documentation for new chunking logic
"""

from rag.main import main

if __name__ == "__main__":
    main()  # Calls the main function from the rag package