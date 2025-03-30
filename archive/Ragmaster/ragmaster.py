# --- rag_app.py ---

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
import os
import threading
import queue
import logging
import sys
import json
import asyncio

# Try to import ThemedTk, but fall back to standard Tk if not available
try:
    from ttkthemes import ThemedTk
    THEMED_TK_AVAILABLE = True
except ImportError:
    THEMED_TK_AVAILABLE = False
    # Log the issue but continue with standard Tk
    print("ttkthemes module not found. Using standard Tkinter without themes.")
    # We'll use tk.Tk instead of ThemedTk

# --- Dependency Imports with Fallbacks ---
# Implement a fallback for dotenv if not available
try:
    from dotenv import load_dotenv, set_key
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("python-dotenv module not found. Using basic environment variable handling.")
    
    # Simple fallback implementations for dotenv functions
    def load_dotenv(dotenv_path=None):
        """Basic implementation to load variables from a .env file"""
        if not dotenv_path or not os.path.exists(dotenv_path):
            return False
        
        with open(dotenv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                key, value = line.split('=', 1)
                os.environ[key] = value
        return True
    
    def set_key(dotenv_path, key, value):
        """Basic implementation to set/update a key in .env file"""
        lines = []
        key_exists = False
        
        # Read existing file
        if os.path.exists(dotenv_path):
            with open(dotenv_path, 'r') as f:
                lines = f.readlines()
        
        # Update or append the key
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{key}="):
                lines[i] = f"{key}={value}\n"
                key_exists = True
                break
        
        if not key_exists:
            lines.append(f"{key}={value}\n")
        
        # Write back to file
        with open(dotenv_path, 'w') as f:
            f.writelines(lines)
        
        # Also set in current environment
        os.environ[key] = value
        return True

# Continue with remaining dependency imports
try:
    import pinecone
    from pinecone import Pinecone, PodSpec, ApiException
    from openai import OpenAI, AsyncOpenAI # Import AsyncOpenAI
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings # Use langchain integration for consistency if preferred
    import PyPDF2
    import docx
    from PIL import Image
    import pytesseract
    import asyncio_tkinter # Import asyncio_tkinter for bridging asyncio and tkinter
except ImportError as e:
    messagebox.showerror(
        "Dependency Error",
        f"Required library not found: {e.name}. "
        "Please install all dependencies using the provided script "
        "(install_dependencies.sh or install_dependencies.bat) "
        "in a virtual environment."
    )
    sys.exit(1)

# --- Constants & Configuration ---
CONFIG_FILE = ".env"
LOG_LEVEL = logging.INFO
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "text-embedding-3-large"
GENERATION_MODEL = "gpt-4o"
PINECONE_UPSERT_BATCH_SIZE = 100 # Pinecone recommended batch size
PINECONE_ENVIRONMENT = "gcp-starter" # Default, change if needed

# --- Logging Setup ---
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
log_queue = queue.Queue() # Queue for thread-safe logging updates to GUI

# --- Helper Functions ---

def load_env_vars():
    """Loads API keys from .env file."""
    if DOTENV_AVAILABLE:
        load_dotenv(dotenv_path=CONFIG_FILE)
    else:
        # Use our fallback implementation
        try:
            load_dotenv(dotenv_path=CONFIG_FILE)
        except Exception as e:
            logging.error(f"Error loading environment variables: {e}")
            
    return os.getenv("PINECONE_API_KEY"), os.getenv("OPENAI_API_KEY")

def save_env_var(key, value):
    """Saves or updates an API key in the .env file."""
    # Create .env if it doesn't exist
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w") as f:
            pass # Just create the file
    
    # Use appropriate function based on availability
    if DOTENV_AVAILABLE:
        set_key(CONFIG_FILE, key, value)
    else:
        # Use our fallback implementation
        try:
            set_key(CONFIG_FILE, key, value)
        except Exception as e:
            logging.error(f"Error saving environment variable: {e}")
            return False
            
    return True

def get_tesseract_path():
    """Tries to find Tesseract path (customize if needed)."""
    # This is a common source of errors. Adjust if Tesseract is installed elsewhere.
    # On Windows, it might be like: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # On Linux/macOS, it's often just 'tesseract' if in PATH.
    if sys.platform.startswith('win'):
        default_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if os.path.exists(default_path):
            return default_path
    # For Linux/macOS, assume it's in PATH
    return 'tesseract'

# Attempt to set tesseract command path globally for pytesseract
try:
    pytesseract.tesseract_cmd = get_tesseract_path()
    # Check if Tesseract is working
    pytesseract.get_tesseract_version()
    logging.info(f"Tesseract found at: {pytesseract.tesseract_cmd}")
except Exception as e:
    logging.warning(f"Tesseract OCR command not found or not working: {e}. OCR functionality will fail. Please install Tesseract and ensure it's in your PATH or update get_tesseract_path().")
    # Don't exit, but OCR won't work. GUI should reflect this later.

# --- Core Logic Classes ---

class DocumentProcessor:
    """Handles text extraction and chunking."""

    def __init__(self, status_callback):
        self.status_callback = status_callback
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )

    def extract_text(self, file_path):
        """Extracts text from various file types."""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        self.status_callback(f"Extracting text from {os.path.basename(file_path)}...")
        logging.info(f"Extracting text from: {file_path}")

        try:
            if ext == '.pdf':
                return self._extract_pdf(file_path)
            elif ext == '.docx':
                return self._extract_docx(file_path)
            elif ext == '.txt':
                return self._extract_txt(file_path)
            elif ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']:
                return self._extract_ocr(file_path)
            else:
                self.status_callback(f"Warning: Unsupported file type '{ext}'. Attempting plain text read.")
                logging.warning(f"Unsupported file type '{ext}'. Attempting plain text read for {file_path}")
                # Try reading as plain text as a fallback
                return self._extract_txt(file_path)
        except Exception as e:
            self.status_callback(f"Error extracting text from {os.path.basename(file_path)}: {e}")
            logging.error(f"Error extracting text from {file_path}: {e}", exc_info=True)
            return None

    def _extract_pdf(self, file_path):
        text = ""
        try:
            # First try text extraction
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                if reader.is_encrypted:
                    try:
                        reader.decrypt('') # Try decrypting with empty password
                    except Exception as decrypt_err:
                         self.status_callback(f"Could not decrypt PDF '{os.path.basename(file_path)}'. Skipping.")
                         logging.warning(f"Could not decrypt {file_path}: {decrypt_err}")
                         return None

                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            # If text is minimal/empty, try OCR as fallback (requires Pillow for pdf2image simulation)
            if len(text.strip()) < 50: # Arbitrary threshold to suspect scanned PDF
                 self.status_callback(f"Minimal text found in PDF '{os.path.basename(file_path)}'. Attempting OCR...")
                 logging.info(f"Minimal text in {file_path}, attempting OCR.")
                 # Simulate pdf2image if library isn't installed or handle differently
                 # For simplicity here, we'll just rely on the initial OCR check in extract_text
                 # A more robust solution might use pdf2image library
                 ocr_text = self._extract_ocr(file_path) # Try direct OCR - might work if tesseract handles PDF
                 if ocr_text and len(ocr_text) > len(text):
                     self.status_callback(f"OCR extracted more text from PDF '{os.path.basename(file_path)}'.")
                     logging.info(f"Using OCR text for PDF {file_path}")
                     return ocr_text

        except Exception as e:
            self.status_callback(f"Error reading PDF {os.path.basename(file_path)}: {e}. Trying OCR.")
            logging.error(f"PyPDF2 error on {file_path}: {e}. Trying OCR.", exc_info=True)
            # Fallback to OCR if direct extraction fails
            ocr_text = self._extract_ocr(file_path)
            if ocr_text:
                return ocr_text
            else:
                 self.status_callback(f"Failed to extract text or OCR from PDF '{os.path.basename(file_path)}'.")
                 logging.error(f"Failed both text extraction and OCR for PDF {file_path}")
                 return None

        return text if text.strip() else None # Return None if only whitespace

    def _extract_docx(self, file_path):
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            self.status_callback(f"Error reading DOCX {os.path.basename(file_path)}: {e}")
            logging.error(f"Error reading DOCX {file_path}: {e}", exc_info=True)
            return None

    def _extract_txt(self, file_path):
        try:
            # Try common encodings
            encodings = ['utf-8', 'latin-1', 'windows-1252']
            text = None
            for enc in encodings:
                try:
                    with open(file_path, 'r', encoding=enc) as f:
                        text = f.read()
                    logging.info(f"Read {file_path} with encoding {enc}")
                    break # Success
                except UnicodeDecodeError:
                    logging.warning(f"Failed to decode {file_path} with {enc}")
                    continue # Try next encoding
            if text is None:
                 raise ValueError("Could not decode file with common encodings.")
            return text
        except Exception as e:
            self.status_callback(f"Error reading TXT {os.path.basename(file_path)}: {e}")
            logging.error(f"Error reading TXT {file_path}: {e}", exc_info=True)
            return None

    def _extract_ocr(self, file_path):
        try:
            # Check if Tesseract command is valid before proceeding
            if not os.path.exists(pytesseract.tesseract_cmd) and sys.platform.startswith('win'):
                 self.status_callback(f"Tesseract not found at '{pytesseract.tesseract_cmd}'. Cannot perform OCR.")
                 logging.error(f"Tesseract not found at specified path '{pytesseract.tesseract_cmd}'.")
                 return None
            if os.path.getsize(file_path) == 0:
                self.status_callback(f"File is empty: {os.path.basename(file_path)}")
                return None

            self.status_callback(f"Performing OCR on {os.path.basename(file_path)}...")
            text = pytesseract.image_to_string(Image.open(file_path))
            self.status_callback(f"OCR complete for {os.path.basename(file_path)}.")
            logging.info(f"OCR successful for {file_path}")
            return text
        except pytesseract.TesseractNotFoundError:
            self.status_callback("Tesseract executable not found. Please install Tesseract OCR and ensure it's in your PATH.")
            logging.error("Tesseract executable not found.", exc_info=True)
            return None
        except FileNotFoundError:
             self.status_callback(f"File not found for OCR: {os.path.basename(file_path)}")
             logging.error(f"File not found during OCR attempt: {file_path}")
             return None
        except Exception as e:
            self.status_callback(f"Error during OCR for {os.path.basename(file_path)}: {e}")
            logging.error(f"Error during OCR for {file_path}: {e}", exc_info=True)
            return None

    def chunk_text(self, text, filename):
        """Chunks the text using the configured splitter."""
        if not text:
            return []
        self.status_callback(f"Chunking text from {filename}...")
        logging.info(f"Chunking text from {filename} (length: {len(text)})...")
        chunks = self.text_splitter.split_text(text)
        # Add metadata (filename) to each chunk
        chunk_data = [{"text": chunk, "metadata": {"source": filename}} for chunk in chunks]
        self.status_callback(f"Created {len(chunks)} chunks for {filename}.")
        logging.info(f"Created {len(chunks)} chunks for {filename}.")
        return chunk_data


class EmbeddingManager:
    """Handles embedding generation using OpenAI."""
    def __init__(self, api_key, status_callback):
        if not api_key:
            raise ValueError("OpenAI API Key is required for EmbeddingManager.")
        # Use AsyncOpenAI for embedding generation
        self.client = AsyncOpenAI(api_key=api_key)
        self.status_callback = status_callback
        self.embedding_model = EMBEDDING_MODEL

    async def get_embeddings(self, chunks_data):
        """Generates embeddings for a list of chunk data asynchronously."""
        if not chunks_data:
            return []

        texts_to_embed = [item["text"] for item in chunks_data]
        if not texts_to_embed:
             return []

        self.status_callback(f"Generating embeddings for {len(texts_to_embed)} chunks...")
        logging.info(f"Requesting embeddings for {len(texts_to_embed)} chunks using {self.embedding_model}...")

        try:
            response = await self.client.embeddings.create(
                input=texts_to_embed,
                model=self.embedding_model
            )
            embeddings = [item.embedding for item in response.data]
            self.status_callback(f"Successfully generated {len(embeddings)} embeddings.")
            logging.info(f"Received {len(embeddings)} embeddings.")

            # Combine embeddings with original chunk data
            for i, item in enumerate(chunks_data):
                 item["embedding"] = embeddings[i]

            return chunks_data # Now contains text, metadata, and embedding

        except Exception as e:
            self.status_callback(f"Error generating embeddings: {e}")
            logging.error(f"OpenAI embedding API error: {e}", exc_info=True)
            return [] # Return empty list on failure


class PineconeManager:
    """Handles interactions with Pinecone vector database."""
    def __init__(self, api_key, status_callback):
        if not api_key:
            raise ValueError("Pinecone API Key is required for PineconeManager.")
        self.status_callback = status_callback
        try:
            self.pinecone_client = Pinecone(api_key=api_key)
            # Check connection by listing indexes (can raise exceptions)
            self.pinecone_client.list_indexes()
            self.status_callback("Pinecone connection successful.")
            logging.info("Pinecone connection successful.")
        except ApiException as e:
             self.status_callback(f"Pinecone API Error: {e.reason} ({e.status}). Check API key and network.")
             logging.error(f"Pinecone API connection error: {e}", exc_info=True)
             raise ConnectionError(f"Failed to connect to Pinecone: {e.reason}") from e
        except Exception as e:
            self.status_callback(f"Pinecone initialization failed: {e}")
            logging.error(f"Pinecone initialization failed: {e}", exc_info=True)
            raise ConnectionError(f"Failed to initialize Pinecone: {e}") from e

    def list_indexes(self):
        """Lists available Pinecone indexes."""
        try:
            indexes = self.pinecone_client.list_indexes().names
            self.status_callback(f"Found indexes: {indexes if indexes else 'None'}")
            logging.info(f"Available Pinecone indexes: {indexes}")
            return indexes if indexes else []
        except ApiException as e:
            self.status_callback(f"Error listing indexes: {e.reason}")
            logging.error(f"Pinecone list_indexes API error: {e}", exc_info=True)
            return []
        except Exception as e:
            self.status_callback(f"Error listing indexes: {e}")
            logging.error(f"Error listing Pinecone indexes: {e}", exc_info=True)
            return []

    def create_index(self, index_name, dimension):
        """Creates a new Pinecone index."""
        if not index_name:
            self.status_callback("Index name cannot be empty.")
            return False
        try:
            self.status_callback(f"Creating index '{index_name}' with dimension {dimension}...")
            logging.info(f"Creating Pinecone index '{index_name}' (dimension: {dimension})...")
            # Use default metric 'cosine' and pod spec
            # NOTE: Adjust 'environment' if you are not using the free tier or a specific cloud provider env
            # Free tier environment is typically automatically assigned now, but 'gcp-starter' or similar might work
            spec = PodSpec(environment=PINECONE_ENVIRONMENT) # Adjust environment if necessary

            self.pinecone_client.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine', # Common choice for OpenAI embeddings
                spec=spec
                )
            self.status_callback(f"Index '{index_name}' created successfully.")
            logging.info(f"Pinecone index '{index_name}' created.")
            return True
        except ApiException as e:
            if e.status == 409: # Conflict - Index already exists
                 self.status_callback(f"Index '{index_name}' already exists.")
                 logging.warning(f"Pinecone index '{index_name}' already exists.")
                 return True # Treat as success if it exists
            else:
                self.status_callback(f"Error creating index '{index_name}': {e.reason} ({e.status})")
                logging.error(f"Pinecone create_index API error for '{index_name}': {e}", exc_info=True)
                return False
        except Exception as e:
            self.status_callback(f"Error creating index '{index_name}': {e}")
            logging.error(f"Error creating Pinecone index '{index_name}': {e}", exc_info=True)
            return False

    def delete_index(self, index_name):
        """Deletes a Pinecone index."""
        if not index_name:
            self.status_callback("Index name cannot be empty.")
            return False
        try:
            self.status_callback(f"Deleting index '{index_name}'...")
            logging.info(f"Deleting Pinecone index '{index_name}'...")
            self.pinecone_client.delete_index(index_name)
            self.status_callback(f"Index '{index_name}' deleted successfully.")
            logging.info(f"Pinecone index '{index_name}' deleted.")
            return True
        except ApiException as e:
             if e.status == 404: # Not Found
                 self.status_callback(f"Index '{index_name}' not found.")
                 logging.warning(f"Attempted to delete non-existent Pinecone index '{index_name}'.")
                 return True # Treat as success if it doesn't exist
             else:
                self.status_callback(f"Error deleting index '{index_name}': {e.reason} ({e.status})")
                logging.error(f"Pinecone delete_index API error for '{index_name}': {e}", exc_info=True)
                return False
        except Exception as e:
            self.status_callback(f"Error deleting index '{index_name}': {e}")
            logging.error(f"Error deleting Pinecone index '{index_name}': {e}", exc_info=True)
            return False

    def list_namespaces(self, index_name):
        """Lists namespaces within an index by fetching stats."""
        if not index_name:
            self.status_callback("Index name required to list namespaces.")
            return []
        try:
            index = self.pinecone_client.Index(index_name)
            stats = index.describe_index_stats()
            namespaces = list(stats.namespaces.keys()) if stats.namespaces else []
            self.status_callback(f"Namespaces in '{index_name}': {namespaces if namespaces else 'None'}")
            logging.info(f"Namespaces in Pinecone index '{index_name}': {namespaces}")
            return namespaces
        except ApiException as e:
             if e.status == 404:
                  self.status_callback(f"Index '{index_name}' not found when listing namespaces.")
                  logging.error(f"Index '{index_name}' not found during namespace listing: {e}")
             else:
                self.status_callback(f"Error listing namespaces for '{index_name}': {e.reason}")
                logging.error(f"Pinecone describe_index_stats API error for '{index_name}': {e}", exc_info=True)
             return []
        except Exception as e:
            self.status_callback(f"Error listing namespaces for '{index_name}': {e}")
            logging.error(f"Error listing Pinecone namespaces for '{index_name}': {e}", exc_info=True)
            return []

    def delete_namespace(self, index_name, namespace):
        """Deletes a namespace within an index."""
        if not index_name or not namespace:
            self.status_callback("Index name and namespace required for deletion.")
            return False
        try:
            index = self.pinecone_client.Index(index_name)
            self.status_callback(f"Deleting namespace '{namespace}' from index '{index_name}'...")
            logging.info(f"Deleting namespace '{namespace}' from Pinecone index '{index_name}'...")
            index.delete(namespace=namespace, delete_all=True)
            self.status_callback(f"Namespace '{namespace}' deleted successfully.")
            logging.info(f"Namespace '{namespace}' deleted from '{index_name}'.")
            return True
        except ApiException as e:
             if e.status == 404:
                  self.status_callback(f"Index '{index_name}' not found when deleting namespace '{namespace}'.")
                  logging.error(f"Index '{index_name}' not found during namespace deletion: {e}")
             else:
                 self.status_callback(f"Error deleting namespace '{namespace}': {e.reason}")
                 logging.error(f"Pinecone delete namespace API error for '{index_name}/{namespace}': {e}", exc_info=True)
             return False
        except Exception as e:
            self.status_callback(f"Error deleting namespace '{namespace}': {e}")
            logging.error(f"Error deleting Pinecone namespace '{namespace}' from '{index_name}': {e}", exc_info=True)
            return False

    def upsert_vectors(self, index_name, namespace, chunks_with_embeddings):
        """Upserts vectors in batches into a Pinecone index/namespace."""
        if not index_name:
            self.status_callback("Index name required for upsert.")
            return False
        if not chunks_with_embeddings:
            self.status_callback("No chunks with embeddings provided to upsert.")
            return False

        try:
            index = self.pinecone_client.Index(index_name)
            total_upserted = 0
            num_chunks = len(chunks_with_embeddings)
            self.status_callback(f"Starting upsert of {num_chunks} vectors into '{index_name}' (namespace: '{namespace or 'default'}')...")
            logging.info(f"Upserting {num_chunks} vectors into '{index_name}/{namespace or 'default'}'.")

            for i in range(0, num_chunks, PINECONE_UPSERT_BATCH_SIZE):
                batch = chunks_with_embeddings[i : i + PINECONE_UPSERT_BATCH_SIZE]
                vectors_to_upsert = []
                for j, item in enumerate(batch):
                    # Create a unique ID for each chunk - important!
                    # Simple strategy: filename + chunk index
                    chunk_id = f"{item['metadata'].get('source', 'unknown')}_{i+j}"
                    vectors_to_upsert.append({
                        "id": chunk_id,
                        "values": item['embedding'],
                        # Store the original text chunk in metadata
                        "metadata": {"text": item['text'], **item['metadata']}
                    })

                if vectors_to_upsert:
                    self.status_callback(f"Upserting batch {i // PINECONE_UPSERT_BATCH_SIZE + 1} ({len(vectors_to_upsert)} vectors)...")
                    logging.debug(f"Upserting batch to {index_name}/{namespace or 'default'}")
                    index.upsert(vectors=vectors_to_upsert, namespace=namespace or None) # Use None for default namespace
                    total_upserted += len(vectors_to_upsert)
                    self.status_callback(f"Upserted {total_upserted}/{num_chunks} vectors...")

            self.status_callback(f"Successfully upserted {total_upserted} vectors into '{index_name}/{namespace or 'default'}'.")
            logging.info(f"Finished upserting {total_upserted} vectors into '{index_name}/{namespace or 'default'}'.")
            return True

        except ApiException as e:
             if e.status == 404:
                  self.status_callback(f"Index '{index_name}' not found during upsert.")
                  logging.error(f"Index '{index_name}' not found during upsert: {e}")
             else:
                self.status_callback(f"Error upserting vectors: {e.reason}")
                logging.error(f"Pinecone upsert API error for '{index_name}/{namespace}': {e}", exc_info=True)
             return False
        except Exception as e:
            self.status_callback(f"Error upserting vectors: {e}")
            logging.error(f"Error upserting vectors to Pinecone '{index_name}/{namespace}': {e}", exc_info=True)
            return False

    def query_vectors(self, index_name, namespace, query_embedding, top_k=5):
        """Queries the Pinecone index for similar vectors."""
        if not index_name:
            self.status_callback("Index name required for query.")
            return []
        if not query_embedding:
            self.status_callback("Query embedding required.")
            return []

        try:
            index = self.pinecone_client.Index(index_name)
            self.status_callback(f"Querying '{index_name}/{namespace or 'default'}'...")
            logging.info(f"Querying Pinecone index '{index_name}/{namespace or 'default'}' (top_k={top_k})...")

            results = index.query(
                vector=query_embedding,
                namespace=namespace or None,
                top_k=top_k,
                include_metadata=True # Crucial to get the text back
            )
            self.status_callback(f"Query returned {len(results.get('matches', []))} results.")
            logging.info(f"Pinecone query returned {len(results.get('matches', []))} matches.")
            # Return the metadata (which includes the text) of the matches
            return results.get('matches', [])

        except ApiException as e:
             if e.status == 404:
                  self.status_callback(f"Index '{index_name}' not found during query.")
                  logging.error(f"Index '{index_name}' not found during query: {e}")
             else:
                self.status_callback(f"Error querying vectors: {e.reason}")
                logging.error(f"Pinecone query API error for '{index_name}/{namespace}': {e}", exc_info=True)
             return []
        except Exception as e:
            self.status_callback(f"Error querying vectors: {e}")
            logging.error(f"Error querying Pinecone '{index_name}/{namespace}': {e}", exc_info=True)
            return []


class RAGSystem:
    """Orchestrates the RAG pipeline."""
    def __init__(self, status_callback, loop): # Add asyncio loop
        self.status_callback = status_callback
        self.loop = loop # Store the asyncio loop
        self.pinecone_api_key = None
        self.openai_api_key = None
        self.doc_processor = DocumentProcessor(status_callback)
        self.embed_manager = None # Initialized when key is set
        self.pinecone_manager = None # Initialized when key is set
        self.openai_client = None # For generation, initialized when key is set

    def configure_keys(self, pinecone_key, openai_key):
        """Sets API keys and initializes dependent managers."""
        # Simple validation
        if not pinecone_key or not openai_key:
             self.status_callback("Error: Both Pinecone and OpenAI API keys are required.")
             return False

        self.pinecone_api_key = pinecone_key
        self.openai_api_key = openai_key
        self.status_callback("API Keys received.")
        logging.info("API Keys configured.")

        try:
            # Initialize managers that depend on keys
            self.status_callback("Initializing Pinecone Manager...")
            self.pinecone_manager = PineconeManager(self.pinecone_api_key, self.status_callback)
            self.status_callback("Initializing OpenAI Embedding Manager...")
            # Pass the asyncio loop to EmbeddingManager if it needs it directly,
            # but here we use the async client which handles its own loop interaction.
            self.embed_manager = EmbeddingManager(self.openai_api_key, self.status_callback)
            # Initialize standard OpenAI client for synchronous generation call (or use async later)
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            self.status_callback("API Clients initialized successfully.")
            logging.info("RAG System API clients initialized.")
            return True
        except ConnectionError as e:
             self.status_callback(f"Configuration failed: {e}")
             logging.error(f"API Client initialization failed: {e}", exc_info=True)
             # Reset managers on failure
             self.pinecone_manager = None
             self.embed_manager = None
             self.openai_client = None
             return False
        except ValueError as e: # Catch key errors from managers
            self.status_callback(f"Configuration failed: {e}")
            logging.error(f"API Key validation failed during manager init: {e}", exc_info=True)
            return False


    def check_dependencies(self):
        """Checks if essential managers are initialized."""
        if not self.pinecone_manager or not self.embed_manager or not self.openai_client:
            self.status_callback("Error: API Keys not configured or initialization failed. Please configure keys first.")
            return False
        return True

    async def process_and_embed_file(self, file_path, index_name, namespace):
        """Processes a single file, embeds chunks, and upserts to Pinecone."""
        if not self.check_dependencies():
            return

        if not index_name:
            self.status_callback("Error: Pinecone index name must be selected.")
            return

        filename = os.path.basename(file_path)
        text = self.doc_processor.extract_text(file_path)
        if not text:
            self.status_callback(f"Could not extract text from {filename}. Skipping.")
            return

        chunks_data = self.doc_processor.chunk_text(text, filename)
        if not chunks_data:
            self.status_callback(f"No chunks created for {filename}. Skipping.")
            return

        # Ensure embedding dimension matches index (if index exists)
        try:
             # Get embedding dimension directly from OpenAI client if possible or assume default
             # For text-embedding-3-large, it's 3072
             embedding_dimension = 3072 # Hardcoded for text-embedding-3-large

             # Check if index exists and create if not
             existing_indexes = self.pinecone_manager.list_indexes()
             if index_name not in existing_indexes:
                  self.status_callback(f"Index '{index_name}' does not exist. Creating...")
                  # Attempt to create the index
                  if not self.pinecone_manager.create_index(index_name, embedding_dimension):
                       self.status_callback(f"Failed to create index '{index_name}'. Aborting upsert.")
                       return
                  # Short pause to allow index creation to propagate
                  await asyncio.sleep(5) # Use asyncio.sleep

             # Optional: Add check here if index *does* exist, verify dimension matches embedding_dimension
             # index_info = self.pinecone_manager.describe_index(index_name)
             # if index_info and index_info.dimension != embedding_dimension:
             #     self.status_callback(f"Error: Index '{index_name}' dimension ({index_info.dimension}) does not match embedding model dimension ({embedding_dimension}).")
             #     return

        except Exception as e:
             self.status_callback(f"Error checking/creating index '{index_name}': {e}")
             logging.error(f"Error during index check/create for {index_name}: {e}", exc_info=True)
             return


        # Get embeddings asynchronously
        chunks_with_embeddings = await self.embed_manager.get_embeddings(chunks_data)

        if not chunks_with_embeddings:
             self.status_callback(f"Failed to generate embeddings for {filename}. Skipping upsert.")
             return

        # Upsert synchronously (or adapt PineconeManager to be async)
        # Running synchronous IO in executor to avoid blocking asyncio loop
        await self.loop.run_in_executor(
             None, # Use default executor (ThreadPoolExecutor)
             self.pinecone_manager.upsert_vectors,
             index_name,
             namespace,
             chunks_with_embeddings
        )
        # Check result if needed, upsert_vectors logs status

    async def answer_query(self, query, index_name, namespace, top_k=5):
        """Handles the RAG process for a given query."""
        if not self.check_dependencies():
            return "Error: System not configured. Set API keys."
        if not query:
            return "Error: Query cannot be empty."
        if not index_name:
             return "Error: Pinecone index name must be selected."

        self.status_callback(f"Processing query: '{query[:50]}...'")

        # 1. Embed the query
        try:
            # Use the async embed manager
            query_embedding_data = await self.embed_manager.get_embeddings([{"text": query, "metadata": {}}]) # Wrap query like a chunk
            if not query_embedding_data or 'embedding' not in query_embedding_data[0]:
                 self.status_callback("Failed to generate query embedding.")
                 return "Error: Could not embed query."
            query_embedding = query_embedding_data[0]['embedding']
        except Exception as e:
            self.status_callback(f"Error embedding query: {e}")
            logging.error(f"Error embedding query: {e}", exc_info=True)
            return "Error: Failed to embed query."

        # 2. Query Pinecone (synchronous, run in executor)
        try:
             matches = await self.loop.run_in_executor(
                  None,
                  self.pinecone_manager.query_vectors,
                  index_name,
                  namespace,
                  query_embedding,
                  top_k
             )
        except Exception as e:
            self.status_callback(f"Error querying Pinecone: {e}")
            logging.error(f"Error querying Pinecone: {e}", exc_info=True)
            return "Error: Failed to query vector database."


        if not matches:
            self.status_callback("No relevant documents found in the vector database for this query.")
            # Optionally, still try to answer with GPT-4o without context
            # return "Could not find relevant context in the database. Please try rephrasing or adding more documents."
            context_str = "No relevant context found."
        else:
            # 3. Format Context
            context_str = "\n---\n".join([match['metadata']['text'] for match in matches if 'metadata' in match and 'text' in match['metadata']])
            self.status_callback(f"Retrieved {len(matches)} context chunks.")

        # 4. Call OpenAI Generation Model (synchronous, run in executor)
        try:
            prompt = f"""Answer the following query based *only* on the provided context. If the context does not contain the answer, state that the information is not available in the provided documents.

            Context:
            {context_str}

            Query: {query}

            Answer:"""

            self.status_callback(f"Generating response using {GENERATION_MODEL}...")
            logging.info(f"Calling {GENERATION_MODEL}...")

            # Run synchronous OpenAI call in executor
            response = await self.loop.run_in_executor(
                 None, # Use default executor
                 self.openai_client.chat.completions.create, # Pass the method itself
                 # Arguments for the method:
                 model=GENERATION_MODEL,
                 messages=[
                      {"role": "system", "content": "You are a helpful assistant answering questions based on provided context."},
                      {"role": "user", "content": prompt}
                 ]
            )

            answer = response.choices[0].message.content
            self.status_callback("Response generated successfully.")
            logging.info(f"{GENERATION_MODEL} response received.")
            return answer.strip()

        except Exception as e:
            self.status_callback(f"Error generating response with OpenAI: {e}")
            logging.error(f"Error calling OpenAI generation API: {e}", exc_info=True)
            return f"Error: Failed to generate response from {GENERATION_MODEL}."


# --- GUI Class ---

class RAG_GUI:
    def __init__(self, root, loop): # Add asyncio loop
        self.root = root
        self.loop = loop # Store the asyncio loop
        self.root.title("Python RAG System")
         # Set minimum size
        self.root.minsize(800, 700)

        # Apply a theme
        self.style = ttk.Style(self.root)
        try:
            # Use a modern theme like 'clam', 'alt', 'default', 'classic'
            # Or from ttkthemes: 'arc', 'equilux', 'itft1', 'plastik', 'radiance', 'ubuntu', etc.
            # self.root is already a ThemedTk instance passed from main
             # Find available themes
             available_themes = self.root.get_themes()
             logging.info(f"Available themes: {available_themes}")
             # Try a few preferred themes
             preferred_themes = ['clam', 'alt', 'default'] # Add 'arc', 'radiance' etc. if ttkthemes works
             for theme in preferred_themes:
                 if theme in available_themes:
                     self.root.set_theme(theme)
                     logging.info(f"Using theme: {theme}")
                     break
             else:
                  logging.warning("Could not set a preferred theme, using default.")

        except Exception as e:
            logging.warning(f"Could not set theme: {e}. Using default Tkinter style.")


        # Initialize RAG System Backend
        self.rag_system = RAGSystem(self.log_status, self.loop) # Pass loop here

        # --- GUI Frames ---
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Configure grid weights for responsiveness
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(0, weight=0) # Config Frame
        self.main_frame.rowconfigure(1, weight=0) # Pinecone Frame
        self.main_frame.rowconfigure(2, weight=1) # Upload Frame
        self.main_frame.rowconfigure(3, weight=1) # Query Frame
        self.main_frame.rowconfigure(4, weight=0) # Status Frame

        # --- Configuration Frame ---
        config_frame = ttk.LabelFrame(self.main_frame, text="API Configuration", padding="10")
        config_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        config_frame.columnconfigure(1, weight=1)

        ttk.Label(config_frame, text="Pinecone API Key:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.pinecone_key_entry = ttk.Entry(config_frame, width=50, show="*")
        self.pinecone_key_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)

        ttk.Label(config_frame, text="OpenAI API Key:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.openai_key_entry = ttk.Entry(config_frame, width=50, show="*")
        self.openai_key_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=2)

        self.save_keys_button = ttk.Button(config_frame, text="Save & Initialize", command=self.save_and_init_keys)
        self.save_keys_button.grid(row=2, column=0, columnspan=2, pady=5)

        # --- Pinecone Management Frame ---
        pinecone_frame = ttk.LabelFrame(self.main_frame, text="Pinecone Management", padding="10")
        pinecone_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        pinecone_frame.columnconfigure(1, weight=1)

        # Index Selection
        ttk.Label(pinecone_frame, text="Index:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.index_var = tk.StringVar()
        self.index_combobox = ttk.Combobox(pinecone_frame, textvariable=self.index_var, state="readonly", postcommand=self.refresh_indexes)
        self.index_combobox.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        self.index_combobox.bind("<<ComboboxSelected>>", self.on_index_select) # Trigger namespace refresh

        # Namespace Selection/Entry
        ttk.Label(pinecone_frame, text="Namespace:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.namespace_var = tk.StringVar()
        self.namespace_combobox = ttk.Combobox(pinecone_frame, textvariable=self.namespace_var) # Allow entry for new namespaces
        self.namespace_combobox.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        # Don't use postcommand here, refresh happens on index select

        # Index Buttons
        index_button_frame = ttk.Frame(pinecone_frame)
        index_button_frame.grid(row=2, column=0, columnspan=2, pady=5)
        self.refresh_indexes_button = ttk.Button(index_button_frame, text="Refresh Indexes", command=self.refresh_indexes)
        self.refresh_indexes_button.pack(side=tk.LEFT, padx=5)
        self.create_index_button = ttk.Button(index_button_frame, text="Create Index", command=self.create_index)
        self.create_index_button.pack(side=tk.LEFT, padx=5)
        self.delete_index_button = ttk.Button(index_button_frame, text="Delete Selected Index", command=self.delete_index)
        self.delete_index_button.pack(side=tk.LEFT, padx=5)

         # Namespace Buttons
        ns_button_frame = ttk.Frame(pinecone_frame)
        ns_button_frame.grid(row=3, column=0, columnspan=2, pady=5)
        self.refresh_ns_button = ttk.Button(ns_button_frame, text="Refresh Namespaces", command=self.refresh_namespaces)
        self.refresh_ns_button.pack(side=tk.LEFT, padx=5)
        self.delete_ns_button = ttk.Button(ns_button_frame, text="Delete Selected Namespace", command=self.delete_namespace)
        self.delete_ns_button.pack(side=tk.LEFT, padx=5)


        # --- Document Upload Frame ---
        upload_frame = ttk.LabelFrame(self.main_frame, text="Document Processing", padding="10")
        upload_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        upload_frame.columnconfigure(0, weight=1)
        upload_frame.rowconfigure(1, weight=1)

        upload_button_frame = ttk.Frame(upload_frame)
        upload_button_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky="ew")

        self.select_files_button = ttk.Button(upload_button_frame, text="Select Files", command=self.select_files)
        self.select_files_button.pack(side=tk.LEFT, padx=5)
        self.select_folder_button = ttk.Button(upload_button_frame, text="Select Folder", command=self.select_folder)
        self.select_folder_button.pack(side=tk.LEFT, padx=5)
        self.process_files_button = ttk.Button(upload_button_frame, text="Process & Upload Selected", command=self.process_selected_files)
        self.process_files_button.pack(side=tk.LEFT, padx=15)

        self.file_listbox = tk.Listbox(upload_frame, selectmode=tk.EXTENDED, width=70, height=10)
        self.file_listbox.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        file_scrollbar = ttk.Scrollbar(upload_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        file_scrollbar.grid(row=1, column=1, sticky="ns")
        self.file_listbox.config(yscrollcommand=file_scrollbar.set)

        self.clear_list_button = ttk.Button(upload_frame, text="Clear List", command=self.clear_file_list)
        self.clear_list_button.grid(row=2, column=0, columnspan=2, pady=5)


        # --- Query Frame ---
        query_frame = ttk.LabelFrame(self.main_frame, text="Query Interface", padding="10")
        query_frame.grid(row=1, column=1, rowspan=3, sticky="nsew", padx=5, pady=5) # Span rows 1, 2, 3
        query_frame.columnconfigure(0, weight=1)
        query_frame.rowconfigure(1, weight=1) # Make response area expand

        ttk.Label(query_frame, text="Enter your query:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.query_entry = ttk.Entry(query_frame, width=60)
        self.query_entry.grid(row=0, column=0, sticky="ew", padx=5, pady=2)
        self.query_entry.bind("<Return>", self.submit_query_event) # Allow Enter key submission

        self.submit_query_button = ttk.Button(query_frame, text="Submit Query", command=self.submit_query)
        self.submit_query_button.grid(row=0, column=1, padx=5, pady=5)

        self.response_text = scrolledtext.ScrolledText(query_frame, wrap=tk.WORD, height=15, state=tk.DISABLED)
        self.response_text.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)


        # --- Status Bar ---
        status_frame = ttk.Frame(self.main_frame, relief=tk.SUNKEN, padding="2")
        status_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Please configure API keys.")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W)
        self.status_label.pack(fill=tk.X)

        # --- Load Initial State ---
        self.load_initial_keys()
        self.update_ui_state() # Disable elements until keys are valid

        # --- Start Log Queue Polling ---
        self.root.after(100, self.process_log_queue) # Check queue every 100ms

    # --- Async Task Handling ---
    def run_async_task(self, coro):
        """Run an async coroutine from the synchronous GUI thread."""
        # Use asyncio_tkinter.run_async to correctly handle asyncio tasks
        asyncio_tkinter.run_async(coro, loop=self.loop)


    def run_sync_in_thread(self, func, *args):
        """Run a synchronous function in a separate thread to avoid blocking GUI."""
        thread = threading.Thread(target=func, args=args, daemon=True)
        thread.start()
        # Optionally, could manage threads more formally if needed (e.g., join on exit)

    # --- GUI Actions ---

    def log_status(self, message):
        """Thread-safe method to update status bar via queue."""
        log_queue.put(message)

    def process_log_queue(self):
        """Processes messages from the log queue to update the status bar."""
        try:
            while True:
                message = log_queue.get_nowait()
                self.status_var.set(message)
                # Optionally, add to a more detailed log window here
        except queue.Empty:
            pass
        finally:
            # Reschedule polling
            self.root.after(100, self.process_log_queue)

    def load_initial_keys(self):
        """Loads keys from .env file on startup."""
        pinecone_key, openai_key = load_env_vars()
        if pinecone_key:
            self.pinecone_key_entry.insert(0, pinecone_key)
        if openai_key:
            self.openai_key_entry.insert(0, openai_key)
        if pinecone_key and openai_key:
             self.log_status("API keys loaded from .env. Initializing...")
             # Automatically initialize if both keys are present
             self.save_and_init_keys()
        else:
             self.log_status("Please enter API keys and click 'Save & Initialize'.")

    def save_and_init_keys(self):
        """Saves keys to .env and initializes the RAG system."""
        pinecone_key = self.pinecone_key_entry.get()
        openai_key = self.openai_key_entry.get()

        if not pinecone_key or not openai_key:
            messagebox.showerror("Missing Keys", "Both Pinecone and OpenAI API keys are required.")
            return

        try:
            save_env_var("PINECONE_API_KEY", pinecone_key)
            save_env_var("OPENAI_API_KEY", openai_key)
            self.log_status("API keys saved to .env file.")
        except Exception as e:
            messagebox.showerror("File Error", f"Could not save API keys to .env file: {e}")
            self.log_status(f"Error saving keys: {e}")
            # Proceed with initialization anyway, using the entered keys
            pass # Allow initialization even if save fails

        self.log_status("Initializing RAG system with provided keys...")
        # Run initialization in a separate thread to avoid blocking GUI
        self.run_sync_in_thread(self._initialize_rag_system, pinecone_key, openai_key)


    def _initialize_rag_system(self, pinecone_key, openai_key):
        """Background task for initializing RAGSystem."""
        success = self.rag_system.configure_keys(pinecone_key, openai_key)
        # Update GUI elements based on success (needs to be scheduled in main thread)
        self.root.after(0, self.update_ui_state)
        if success:
            # Refresh indexes after successful initialization
            self.root.after(0, self.refresh_indexes)

    def update_ui_state(self):
        """Enables/Disables GUI elements based on RAG system readiness."""
        initialized = self.rag_system.check_dependencies()
        state = tk.NORMAL if initialized else tk.DISABLED

        # Pinecone controls
        self.index_combobox.config(state=tk.NORMAL if initialized else tk.DISABLED) # Enable even if disabled before list populates
        self.namespace_combobox.config(state=tk.NORMAL if initialized else tk.DISABLED)
        self.refresh_indexes_button.config(state=state)
        self.create_index_button.config(state=state)
        self.delete_index_button.config(state=state)
        self.refresh_ns_button.config(state=state)
        self.delete_ns_button.config(state=state)

        # Document processing controls
        self.select_files_button.config(state=state)
        self.select_folder_button.config(state=state)
        self.process_files_button.config(state=state)
        # self.clear_list_button state is always NORMAL

        # Query controls
        self.query_entry.config(state=state)
        self.submit_query_button.config(state=state)

    def refresh_indexes(self):
        """Fetches and updates the list of Pinecone indexes."""
        if not self.rag_system.check_dependencies():
            self.log_status("Cannot refresh indexes: System not initialized.")
            return [] # Return empty list for postcommand

        self.log_status("Refreshing Pinecone indexes...")
        # Run in thread to avoid blocking GUI
        self.run_sync_in_thread(self._fetch_indexes)
        return [] # Important for postcommand, it expects a return value

    def _fetch_indexes(self):
        """Background task to fetch indexes."""
        indexes = self.rag_system.pinecone_manager.list_indexes()
        # Schedule GUI update in main thread
        self.root.after(0, self._update_index_combobox, indexes)

    def _update_index_combobox(self, indexes):
         """Updates the index combobox in the main thread."""
         current_index = self.index_var.get()
         self.index_combobox['values'] = indexes
         if indexes:
             if current_index in indexes:
                 self.index_var.set(current_index) # Keep selection if still valid
             else:
                 self.index_var.set(indexes[0]) # Default to first index
                 self.on_index_select() # Trigger namespace refresh for new default index
         else:
             self.index_var.set("") # Clear if no indexes
             self.namespace_combobox['values'] = [] # Clear namespaces too
             self.namespace_var.set("")
         self.log_status("Index list updated.")


    def on_index_select(self, event=None):
        """Handler when an index is selected."""
        self.refresh_namespaces() # Refresh namespaces for the newly selected index

    def refresh_namespaces(self):
        """Fetches and updates namespaces for the selected index."""
        index_name = self.index_var.get()
        if not self.rag_system.check_dependencies():
            self.log_status("Cannot refresh namespaces: System not initialized.")
            return
        if not index_name:
            # Clear namespaces if no index is selected
            self.namespace_combobox['values'] = []
            self.namespace_var.set("")
            # self.log_status("Select an index to see namespaces.")
            return

        self.log_status(f"Refreshing namespaces for index '{index_name}'...")
        self.run_sync_in_thread(self._fetch_namespaces, index_name)

    def _fetch_namespaces(self, index_name):
        """Background task to fetch namespaces."""
        namespaces = self.rag_system.pinecone_manager.list_namespaces(index_name)
        self.root.after(0, self._update_namespace_combobox, namespaces)

    def _update_namespace_combobox(self, namespaces):
        """Updates the namespace combobox in the main thread."""
        current_ns = self.namespace_var.get()
        # Ensure 'Default' (empty string) is always an option if allowing default namespace usage
        display_namespaces = [""] + namespaces # Add empty string for default namespace
        self.namespace_combobox['values'] = display_namespaces
        if current_ns in display_namespaces:
             self.namespace_var.set(current_ns) # Keep selection if valid
        elif namespaces: # If namespaces exist but current isn't one, default to empty string (default ns)
             self.namespace_var.set("")
        else: # No namespaces exist, default to empty string
             self.namespace_var.set("")
        self.log_status("Namespace list updated.")


    def create_index(self):
        """Prompts user for new index name and creates it."""
        if not self.rag_system.check_dependencies():
            messagebox.showwarning("Not Ready", "System not initialized. Configure API keys first.")
            return

        index_name = simpledialog.askstring("Create Index", "Enter new index name:", parent=self.root)
        if index_name:
             # Assume dimension based on the configured embedding model
             # For text-embedding-3-large, dimension is 3072
             dimension = 3072
             self.log_status(f"Requesting creation of index '{index_name}' (dim: {dimension})...")
             # Run in thread
             self.run_sync_in_thread(self._create_index_task, index_name, dimension)
        else:
             self.log_status("Index creation cancelled.")

    def _create_index_task(self, index_name, dimension):
        """Background task for index creation."""
        success = self.rag_system.pinecone_manager.create_index(index_name, dimension)
        if success:
            # Refresh index list after creation
            self.root.after(0, self.refresh_indexes)

    def delete_index(self):
        """Deletes the currently selected index after confirmation."""
        index_name = self.index_var.get()
        if not index_name:
            messagebox.showwarning("No Index", "Please select an index to delete.")
            return
        if not self.rag_system.check_dependencies():
            messagebox.showwarning("Not Ready", "System not initialized.")
            return

        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to permanently delete the index '{index_name}' and all its data?", parent=self.root):
             self.log_status(f"Requesting deletion of index '{index_name}'...")
             # Run in thread
             self.run_sync_in_thread(self._delete_index_task, index_name)
        else:
             self.log_status("Index deletion cancelled.")

    def _delete_index_task(self, index_name):
         """Background task for index deletion."""
         success = self.rag_system.pinecone_manager.delete_index(index_name)
         if success:
             # Refresh index list after deletion
             self.root.after(0, self.refresh_indexes)


    def delete_namespace(self):
        """Deletes the selected namespace from the selected index."""
        index_name = self.index_var.get()
        namespace = self.namespace_var.get()

        if not index_name:
             messagebox.showwarning("No Index", "Please select an index first.")
             return
        if namespace == "": # User selected the 'Default' placeholder
            messagebox.showinfo("Default Namespace", "Cannot explicitly delete the default namespace. Upsert without specifying a namespace to use it.")
            return
        if not namespace: # Should not happen if default is handled, but check anyway
            messagebox.showwarning("No Namespace", "Please select or enter a namespace to delete.")
            return
        if not self.rag_system.check_dependencies():
             messagebox.showwarning("Not Ready", "System not initialized.")
             return

        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to permanently delete namespace '{namespace}' from index '{index_name}'?", parent=self.root):
            self.log_status(f"Requesting deletion of namespace '{namespace}' from '{index_name}'...")
            # Run in thread
            self.run_sync_in_thread(self._delete_namespace_task, index_name, namespace)
        else:
            self.log_status("Namespace deletion cancelled.")

    def _delete_namespace_task(self, index_name, namespace):
        """Background task for namespace deletion."""
        success = self.rag_system.pinecone_manager.delete_namespace(index_name, namespace)
        if success:
            # Refresh namespace list after deletion
            self.root.after(0, self.refresh_namespaces)


    def select_files(self):
        """Opens file dialog to select multiple files."""
        filetypes = [
            ("Supported Files", "*.pdf *.docx *.txt *.jpg *.jpeg *.png *.tiff *.bmp *.gif"),
            ("PDF Files", "*.pdf"),
            ("Word Documents", "*.docx"),
            ("Text Files", "*.txt"),
            ("Image Files", "*.jpg *.jpeg *.png *.tiff *.bmp *.gif"),
            ("All Files", "*.*")
        ]
        files = filedialog.askopenfilenames(title="Select Files", filetypes=filetypes, parent=self.root)
        if files:
            current_files = set(self.file_listbox.get(0, tk.END))
            new_files_added = 0
            for f in files:
                 if f not in current_files:
                     self.file_listbox.insert(tk.END, f)
                     new_files_added += 1
            self.log_status(f"Added {new_files_added} new file(s) to the list.")

    def select_folder(self):
        """Opens folder dialog and adds supported files recursively."""
        folder = filedialog.askdirectory(title="Select Folder Containing Documents", parent=self.root)
        if folder:
            self.log_status(f"Scanning folder '{folder}'...")
            self.run_sync_in_thread(self._scan_folder_task, folder)

    def _scan_folder_task(self, folder):
        """Background task to scan folder for supported files."""
        supported_extensions = ['.pdf', '.docx', '.txt', '.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']
        found_files = []
        for root_dir, _, filenames in os.walk(folder):
            for filename in filenames:
                if any(filename.lower().endswith(ext) for ext in supported_extensions):
                    found_files.append(os.path.join(root_dir, filename))

        # Update listbox in main thread
        self.root.after(0, self._add_files_to_listbox, found_files)


    def _add_files_to_listbox(self, files_to_add):
         """Adds found files to listbox in main thread."""
         current_files = set(self.file_listbox.get(0, tk.END))
         new_files_added = 0
         for f in files_to_add:
             if f not in current_files:
                 self.file_listbox.insert(tk.END, f)
                 new_files_added += 1
         self.log_status(f"Added {new_files_added} file(s) from folder scan.")


    def clear_file_list(self):
        """Clears the file selection listbox."""
        self.file_listbox.delete(0, tk.END)
        self.log_status("File list cleared.")

    def process_selected_files(self):
        """Processes the files selected in the listbox."""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Files Selected", "Please select files from the list to process.")
            return

        files_to_process = [self.file_listbox.get(i) for i in selected_indices]
        index_name = self.index_var.get()
        namespace = self.namespace_var.get() # Can be empty string for default

        if not index_name:
             messagebox.showerror("No Index", "Please select a Pinecone index first.")
             return
        if not self.rag_system.check_dependencies():
            messagebox.showwarning("Not Ready", "System not initialized.")
            return

        self.log_status(f"Starting processing for {len(files_to_process)} files...")
        # Disable button during processing
        self.process_files_button.config(state=tk.DISABLED)

        # Run the async processing task
        self.run_async_task(self._process_files_async(files_to_process, index_name, namespace))


    async def _process_files_async(self, files, index_name, namespace):
        """Asynchronous task to process multiple files."""
        total_files = len(files)
        processed_count = 0
        for file_path in files:
            processed_count += 1
            self.log_status(f"Processing file {processed_count}/{total_files}: {os.path.basename(file_path)}...")
            try:
                # Await the processing, embedding, and upserting for each file
                await self.rag_system.process_and_embed_file(file_path, index_name, namespace)
            except Exception as e:
                 # Log errors but continue with the next file
                 self.log_status(f"Failed processing {os.path.basename(file_path)}: {e}")
                 logging.error(f"Error during async processing of {file_path}: {e}", exc_info=True)

        self.log_status(f"Finished processing {total_files} selected files.")
        # Re-enable button (schedule in main thread)
        self.root.after(0, lambda: self.process_files_button.config(state=tk.NORMAL))


    def submit_query_event(self, event):
         """Handles query submission via Enter key."""
         self.submit_query()

    def submit_query(self):
        """Submits the query to the RAG system."""
        query = self.query_entry.get()
        index_name = self.index_var.get()
        namespace = self.namespace_var.get()

        if not query:
            messagebox.showwarning("Empty Query", "Please enter a query.")
            return
        if not index_name:
             messagebox.showerror("No Index", "Please select a Pinecone index first.")
             return
        if not self.rag_system.check_dependencies():
            messagebox.showwarning("Not Ready", "System not initialized.")
            return

        self.log_status("Submitting query...")
        self.response_text.config(state=tk.NORMAL)
        self.response_text.delete(1.0, tk.END)
        self.response_text.insert(tk.END, "Processing your query...\n")
        self.response_text.config(state=tk.DISABLED)
        self.submit_query_button.config(state=tk.DISABLED)
        self.query_entry.config(state=tk.DISABLED)

        # Run the async query task
        self.run_async_task(self._answer_query_async(query, index_name, namespace))

    async def _answer_query_async(self, query, index_name, namespace):
        """Asynchronous task to get answer for a query."""
        try:
            answer = await self.rag_system.answer_query(query, index_name, namespace)
        except Exception as e:
            answer = f"An unexpected error occurred: {e}"
            logging.error(f"Unexpected error during query processing: {e}", exc_info=True)

        # Schedule GUI update in main thread
        self.root.after(0, self._update_response_text, answer)

    def _update_response_text(self, answer):
        """Updates the response text area in the main thread."""
        self.response_text.config(state=tk.NORMAL)
        self.response_text.delete(1.0, tk.END) # Clear "Processing..."
        self.response_text.insert(tk.END, answer)
        self.response_text.config(state=tk.DISABLED)
        self.log_status("Query finished.")
        # Re-enable query input
        self.submit_query_button.config(state=tk.NORMAL)
        self.query_entry.config(state=tk.NORMAL)


# --- Main Execution ---
def main():
    # Create the main window, using ThemedTk if available, otherwise standard Tk
    if THEMED_TK_AVAILABLE:
        root = ThemedTk(theme="clam")  # Initialize with a theme
    else:
        root = tk.Tk()  # Fallback to standard Tk
        root.title("Python RAG System")  # We need to set the title explicitly with standard Tk

    # Get the asyncio event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Initialize the GUI class, passing the root window and the loop
    app = RAG_GUI(root, loop)

    # Define how to close the application gracefully
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            logging.info("Closing application.")
            loop.call_soon_threadsafe(loop.stop) # Stop the asyncio loop
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Start the Tkinter main loop using asyncio_tkinter bridge
    try:
        # Start the asyncio loop in a separate thread managed by asyncio_tkinter
        # Start Tkinter's main loop
        asyncio_tkinter.mainloop(loop)
    except KeyboardInterrupt:
        logging.info("Application interrupted.")
        if not loop.is_closed():
            loop.call_soon_threadsafe(loop.stop)
    finally:
        if not loop.is_closed():
             # Ensure loop stops cleanly if not already stopped
             # Running loop.stop() directly here might cause issues if called from wrong thread
             # loop.call_soon_threadsafe(loop.stop) is generally safer
             pass # asyncio_tkinter.mainloop should handle loop cleanup

        logging.info("Application closed.")


if __name__ == "__main__":
    main()

