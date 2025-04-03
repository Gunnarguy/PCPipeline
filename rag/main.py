# cSpell:ignore dotenv levelname
import sys
import os
import logging
from dotenv import load_dotenv

# Better path handling for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Load environment variables from .env file
load_dotenv()


# Try to initialize the system, with better error handling
def main() -> None:
    """Start Retrieval Augmented Generation system."""
    try:
        # Configure logging first
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler()]
        )
        # Import here to avoid circular imports
        try:
            from rag.processing import init_system
            init_system()

            # Check if Pinecone was initialized successfully
            from rag.processing import pc
            if pc is None:
                print("\nERROR: Failed to initialize Pinecone.")
                print("Make sure your PINECONE_API_KEY environment "
                      "variable is set")
                print("correctly.")
                print("Try creating a .env file in the project root with:")
                print("PINECONE_API_KEY=your-api-key")
                print("OPENAI_API_KEY=your-api-key")
                sys.exit(1)
            # If we got here, continue with the application
            from rag.gui import GuidedRAGInterface
            app = GuidedRAGInterface()
            app.mainloop()
        except ModuleNotFoundError as e:
            print(f"\nModule import error: {e}")
            print("Make sure the 'rag' package exists and is properly "
                  "installed.")
            print("Try running: pip install -e .")
            sys.exit(1)
    except ImportError as e:
        if "pinecone" in str(e).lower():
            print("\nERROR: Problem with Pinecone package.")
            print("Please install Pinecone SDK with:")
            print("  pip install pinecone")
            sys.exit(1)
        else:
            print(f"Import error: {e}")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Initialization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
