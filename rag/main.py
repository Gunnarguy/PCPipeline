import sys
import os

# Add the parent directory to sys.path so we can use absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to initialize the system, with better error handling
def main() -> None:
    """Start Retrieval Augmented Generation system."""
    try:
        # Import here to avoid circular imports
        from rag.processing import init_system, pc_index
        init_system()
        
        # Check if Pinecone was initialized successfully
        from rag.processing import pc
        if pc is None:
            print("\nERROR: Failed to initialize Pinecone.")
            print("Please run the Pinecone installer script:")
            print("  python /Users/gunnarhostetler/Documents/GitHub/PCPipeline/install_pinecone.py")
            sys.exit(1)
            
        # If we got here, continue with the application
        from rag.gui import GuidedRAGInterface
        app = GuidedRAGInterface()
        app.mainloop()
    except ImportError as e:
        if "pinecone" in str(e).lower():
            print("\nERROR: Problem with Pinecone package.")
            print("Please run the Pinecone installer script:")
            print("  python /Users/gunnarhostetler/Documents/GitHub/PCPipeline/install_pinecone.py")
            sys.exit(1)
        else:
            print(f"Import error: {e}")
            sys.exit(1)
    except Exception as e:
        sys.exit(f"Initialization failed: {e}")

if __name__ == "__main__":
    main()
