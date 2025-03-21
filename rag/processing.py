import os, sys, time, hashlib, logging, traceback
import threading, queue
from pathlib import Path
from typing import Any, List, Optional, Dict, Tuple, Union, Generator
import magic, textract, fitz, pytesseract
from PIL import Image
import torch
import pinecone
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv
from rag.config import Config
import warnings
import tiktoken  # For better token counting visualization
import pandas as pd  # For better data handling during processing

# Suppress specific transformers warnings
warnings.filterwarnings("ignore", message=".*BatchEncoding.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Global variables that will be available for import
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()
oai_client = None
pc = None  # Initialize pc at module level so it can be imported
reranker = None
reranker_tokenizer = None  # Add a separate variable for the tokenizer
device = None
pc_index = None  # Make pc_index available at module level too
_initialized = False  # Track whether initialization has been done

def init_system() -> None:
    """Initialize API clients, Pinecone, and ML models."""
    global _initialized, oai_client, pc, reranker, reranker_tokenizer, device
    
    # Skip re-initialization if already done
    if _initialized:
        logging.debug("System already initialized, skipping re-initialization")
        return
    
    if not os.getenv("PINECONE_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        sys.exit("Missing required API key(s): Check PINECONE_API_KEY and OPENAI_API_KEY")
    
    # Initialize OpenAI client
    try:
        oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Test the client with a simple completion to ensure it works
        oai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Test"}],
            max_tokens=5
        )
        logging.info("OpenAI client initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
        oai_client = None  # Make sure it's explicitly None if initialization fails
    
    # Initialize Pinecone
    try:
        pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        logging.info("Pinecone client initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize Pinecone: {e}")
        pc = None

    # Initialize device and reranker
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Initialize reranker model and tokenizer separately
        reranker = AutoModelForSequenceClassification.from_pretrained(Config.RERANKER_MODEL).to(device)
        reranker_tokenizer = AutoTokenizer.from_pretrained(Config.RERANKER_MODEL)
        # Make sure we mark that initialization succeeded
        logging.info(f"Successfully loaded reranker model {Config.RERANKER_MODEL}")
    except Exception as e:
        logging.error(f"Failed to load reranker model: {e}")
        reranker = None
        reranker_tokenizer = None
        
    _initialized = True  # Mark as initialized

def wait_for_index_ready(index_name: str, timeout: int = 60, poll_interval: int = 1) -> bool:
    """Wait until the specified index is ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            desc = pc.describe_index(index_name)
            if desc.get("status", {}).get("state") == "Ready":
                return True
        except pinecone.core.client.exceptions.NotFoundException:
            pass
        time.sleep(poll_interval)
    return False

def extract_index_names(raw_indexes: Any) -> List[str]:
    """
    Extract a list of index names from the Pinecone list_indexes() response.
    Handles various response formats.
    """
    try:
        if hasattr(raw_indexes, 'names'):
            return raw_indexes.names()
    except (AttributeError, TypeError):
        pass

    if isinstance(raw_indexes, dict) and "indexes" in raw_indexes:
        descs = raw_indexes["indexes"]
        if descs and isinstance(descs[0], dict) and "name" in descs[0]:
            return [d["name"] for d in descs]
        else:
            return []
    elif isinstance(raw_indexes, list):
        if not raw_indexes:
            return []
        first = raw_indexes[0]
        if isinstance(first, dict) and "name" in first:
            return [d["name"] for d in raw_indexes]
        elif isinstance(first, str):
            return raw_indexes
        else:
            return []
    elif isinstance(raw_indexes, str):
        return [raw_indexes]
    else:
        return []

# -----------------------------
# File and Text Processors
# -----------------------------
class FileProcessor:
    """Class to handle file processing based on MIME type."""

    def __init__(self) -> None:
        self.textract_config = {
            'pdftotext': {'layout': True},
            'ocr': {'language': 'eng'}
        }

    def process_file(self, path: Path) -> Optional[Tuple[Optional[str], Optional[str]]]: # Return text and mime
        """Process a file based on its MIME type."""
        try:
            mime = magic.from_file(str(path), mime=True)
            if mime not in Config.ACCEPTED_MIME_TYPES:
                logging.info(f"Skipping unsupported MIME type: {mime}")
                return None, mime # Return None for text, but still mime type
            text = self._dispatch_processing(path, mime)
            return text, mime
        except Exception as e:
            logging.error(f"Error processing {path}: {e}")
            return None, None

    def _dispatch_processing(self, path: Path, mime: str) -> Optional[str]:
        """Dispatch file processing based on MIME type."""
        if mime == 'application/pdf':
            return self._process_pdf(path)
        elif mime.startswith('image/'):
            return self._process_image(path)
        elif mime == 'application/octet-stream':
            return self._process_octet_stream(path)
        else:
            return self._process_generic(path)
    
    def _process_octet_stream(self, path: Path) -> Optional[str]:
        """Handle generic binary files by attempting multiple extraction methods."""
        # Try to infer a more specific type from the file extension
        suffix = path.suffix.lower()
        
        # Check if it might be a document file based on extension
        document_exts = ['.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.pdf']
        if suffix in document_exts:
            try:
                return textract.process(str(path), **self.textract_config).decode('utf-8')
            except Exception as e:
                logging.debug(f"Textract failed on binary file {path}: {e}")
        
        # Try as PDF if extension suggests it
        if suffix == '.pdf':
            try:
                return self._process_pdf(path)
            except Exception as e:
                logging.debug(f"PDF processing failed on binary file {path}: {e}")
        
        # Try as an image if extension suggests it
        image_exts = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
        if suffix in image_exts:
            try:
                return self._process_image(path)
            except Exception as e:
                logging.debug(f"Image processing failed on binary file {path}: {e}")
        
        # As a last resort, try textract generically
        try:
            return textract.process(str(path)).decode('utf-8', errors='ignore')
        except Exception:
            # If all attempts failed, log at debug level instead of warning or error
            logging.debug(f"All extraction methods failed for binary file {path}")
            return None

    def _process_pdf(self, path: Path) -> Optional[str]:
        """Extract text from PDF using PyMuPDF and OCR fallback, now with structured extraction."""
        text = ""
        try:
            structured_chunks = self._process_pdf_with_structure(path) # Use structured processing
            for chunk_data in structured_chunks:
                text += chunk_data["text"] + "\n" # Reconstruct text from structured chunks
            return text
        except Exception as e:
            logging.error(f"PDF processing error ({path}): {e}")
            return None

    def _process_pdf_with_structure(self, path: Path) -> List[Dict]:
        """Extract text from PDF with structural information preserved."""
        structured_chunks = []
        try:
            with fitz.open(str(path)) as doc:
                for page_num, page in enumerate(doc):
                    blocks = page.get_text("dict")["blocks"]
                    for block in blocks:
                        if block["type"] == 0:  # Text block
                            block_text = ""
                            for line in block["lines"]:
                                block_text += "".join([span["text"] for span in line["spans"]]) + "\n"

                            font_size = block["lines"][0]["spans"][0]["size"] if block["lines"] and block["lines"][0]["spans"] else 12 # Default font size
                            is_heading = font_size > 14  # Heuristic for heading detection, adjust as needed

                            structured_chunks.append({
                                "text": block_text.strip(),
                                "page": page_num + 1,
                                "is_heading": is_heading,
                                "font_size": font_size,
                                "position": block["bbox"]
                            })
            return structured_chunks
        except Exception as e:
            logging.error(f"Structured PDF processing error ({path}): {e}")
            return []

    def _process_image(self, path: Path) -> Optional[str]:
        """Extract text from an image using Tesseract OCR."""
        try:
            return pytesseract.image_to_string(
                Image.open(path),
                config='--psm 6 -c preserve_interword_spaces=1'
            )
        except Exception as e:
            logging.error(f"Image processing error ({path}): {e}")
            return None

    def _process_generic(self, path: Path) -> Optional[str]:
        """Extract text using textract; fallback to direct file reading."""
        try:
            return textract.process(str(path), **self.textract_config).decode('utf-8')
        except Exception as e:
            logging.error(f"Text extraction error ({path}): {e}")
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except Exception:
                return None

class TextProcessor:
    """Class for text splitting and chunking with detailed analytics."""

    def __init__(self) -> None:
        self.tokenizers = {}
        self._init_tokenizers()
        
    def _init_tokenizers(self):
        """Initialize tokenizers for various models."""
        try:
            self.tokenizers["cl100k_base"] = tiktoken.get_encoding("cl100k_base")  # GPT-4, text-embedding-3-*
            self.tokenizers["p50k_base"] = tiktoken.get_encoding("p50k_base")  # GPT-3.5, text-embedding-ada-002
        except Exception as e:
            logging.warning(f"Could not initialize tiktoken encoders: {e}")
        
    def get_tokenizer(self, model: str = "cl100k_base") -> Optional[Any]:
        """Get the appropriate tokenizer for a model."""
        return self.tokenizers.get(model)

    def count_tokens(self, text: str, model: str = "cl100k_base") -> int:
        """Count tokens for a piece of text using the specified model's tokenizer."""
        tokenizer = self.get_tokenizer(model)
        if tokenizer:
            return len(tokenizer.encode(text))
        # Fallback approximation if tokenizer not available
        return len(text) // 4  # Rough estimate: ~4 chars per token

    def get_splitter_for_mimetype(self, mime_type: str) -> RecursiveCharacterTextSplitter:
        """Dynamically determine text splitter based on MIME type."""
        if mime_type == 'application/pdf':
            return RecursiveCharacterTextSplitter(
                chunk_size=1200,  # Slightly larger for PDFs to keep sections together
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""], # More separators for PDFs
                length_function=len
            )
        elif mime_type.startswith('text/plain') or mime_type in ['text/markdown', 'text/rtf', 'application/rtf', 'text/csv', 'text/tsv']:
            return RecursiveCharacterTextSplitter(
                chunk_size=800,    # Smaller chunks for plain text
                chunk_overlap=150,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len
            )
        elif mime_type in ['application/x-python', 'text/x-python', 'application/javascript', 'text/javascript', 'text/css']: # Code types
            return TokenTextSplitter( # Use token splitter for code
                chunk_size=500,      # Smaller chunks for code
                chunk_overlap=50,
            )
        elif mime_type == 'text/html':
            return RecursiveCharacterTextSplitter(
                chunk_size=1000,   # HTML might need slightly larger context
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len
            )
        else: # Default for other text-based types and fallbacks
            return RecursiveCharacterTextSplitter(
                chunk_size=Config.DEFAULT_CHUNK_SIZE,
                chunk_overlap=Config.DEFAULT_CHUNK_OVERLAP,
                length_function=len
            )

    def chunk_text(self, text: str, metadata: dict, mime_type: str) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Split text into chunks and attach content hash, with detailed analytics.
        Returns both chunks and statistics about the chunking process.
        """
        if not text:
            return [], {"total_chunks": 0, "total_tokens": 0, "avg_tokens_per_chunk": 0}
            
        # Get appropriate splitter for this content type
        splitter = self.get_splitter_for_mimetype(mime_type)
        
        # Create chunks
        chunks = splitter.create_documents([text], [metadata])
        
        # Initialize analytics dict
        analytics = {
            "total_chunks": len(chunks),
            "total_tokens": 0,
            "token_distribution": [],
            "chunk_sizes": [],
            "mime_type": mime_type,
            "chunk_strategy": str(splitter.__class__.__name__),
        }
        
        # Process each chunk with analytics and add hash
        for chunk in chunks:
            # Add hash to metadata
            chunk.metadata['content_hash'] = hashlib.sha256(chunk.page_content.encode()).hexdigest()
            
            # Count tokens using tiktoken if available
            token_count = self.count_tokens(chunk.page_content)
            analytics["token_distribution"].append(token_count)
            analytics["total_tokens"] += token_count
            analytics["chunk_sizes"].append(len(chunk.page_content))
        
        # Calculate averages
        if chunks:
            analytics["avg_tokens_per_chunk"] = analytics["total_tokens"] / len(chunks)
            analytics["avg_chars_per_chunk"] = sum(analytics["chunk_sizes"]) / len(chunks)
            analytics["min_tokens"] = min(analytics["token_distribution"]) if analytics["token_distribution"] else 0
            analytics["max_tokens"] = max(analytics["token_distribution"]) if analytics["token_distribution"] else 0
        
        return chunks, analytics

    def semantic_chunk(self, text: str, metadata: dict, mime_type: str) -> List[Any]: # Placeholder for semantic chunking
        """Placeholder for future semantic chunking implementation."""
        # In a real implementation, use embeddings to group semantically similar chunks
        logging.info("Semantic chunking placeholder - using standard chunking instead.")
        return self.chunk_text(text, metadata, mime_type) # Fallback to standard chunking

    def determine_optimal_chunk_parameters(self, query: str) -> Tuple[int, int]: # Placeholder for query-aware chunking
        """Placeholder for future query-aware chunk parameter determination."""
        # Analyze query to adjust chunk size/overlap dynamically
        logging.info("Query-aware chunking parameter determination placeholder - using defaults.")
        return Config.DEFAULT_CHUNK_SIZE, Config.DEFAULT_CHUNK_OVERLAP # Return default values for now

    def visualize_chunk(self, text: str, model: str = "cl100k_base") -> Dict[str, Any]:
        """Create a visualization of how text would be tokenized and chunked."""
        tokenizer = self.get_tokenizer(model)
        if not tokenizer:
            return {"error": "Tokenizer not available"}
            
        tokens = tokenizer.encode(text)
        decoded = [tokenizer.decode_single_token_bytes(t).decode('utf-8', errors='replace') for t in tokens]
        
        return {
            "original_text": text,
            "token_count": len(tokens),
            "tokens": decoded,
            "character_count": len(text),
            "tokens_per_char": len(tokens) / len(text) if text else 0,
        }

def generate_embeddings(texts: List[str]) -> Optional[List[Any]]:
    """Generate embeddings using the OpenAI client."""
    try:
        response = oai_client.embeddings.create(
            input=texts,
            model=Config.EMBEDDING_MODEL,
            dimensions=Config.EMBEDDING_DIM
        )
        return [e.embedding for e in response.data]
    except Exception as e:
        logging.error(f"Embedding error: {e}")
        return None