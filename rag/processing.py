import hashlib
import logging
import os
import sys
import time

# import threading, queue # Unused
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Re-added Union for type unions; Generator omitted
import fitz
import magic
import pytesseract

try:
    # import textract # Already commented out
    # HAS_TEXTRACT = True
    HAS_TEXTRACT = False  # Set to False since textract is commented out
except ImportError:
    HAS_TEXTRACT = False
    print(
        "Warning: textract module not available. "
        "Some document types may not be processed correctly."
    )
import warnings

import pinecone
import tiktoken
import torch
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from openai import OpenAI
from PIL import Image
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rag.config import Config

# import pandas as pd # Unused

# Suppress specific transformers warnings
warnings.filterwarnings("ignore", message=".*BatchEncoding.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
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
        sys.exit(
            "Missing required API key(s): Check PINECONE_API_KEY and OPENAI_API_KEY"
        )

    # Initialize OpenAI client
    try:
        oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Test the client with a simple completion to ensure it works
        oai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Test"}],
            max_tokens=5,
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
        reranker = AutoModelForSequenceClassification.from_pretrained(
            Config.RERANKER_MODEL
        ).to(device)
        reranker_tokenizer = AutoTokenizer.from_pretrained(Config.RERANKER_MODEL)
        # Mark that initialization succeeded
        logging.info(f"Successfully loaded reranker model {Config.RERANKER_MODEL}")
    except Exception as e:
        logging.error(f"Failed to load reranker model: {e}")
        reranker = None
        reranker_tokenizer = None

    _initialized = True  # Mark as initialized


def wait_for_index_ready(
    index_name: str, timeout: int = 60, poll_interval: int = 1
) -> bool:
    """Wait until the specified index is ready."""
    if pc is None:
        return False
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            desc = pc.describe_index(index_name)
            if desc.get("status", {}).get("state") == "Ready":
                return True
        except pinecone.core.client.exceptions.NotFoundException:  # type: ignore
            pass
        time.sleep(poll_interval)
    return False


def extract_index_names(raw_indexes: Any) -> List[str]:
    """
    Extract a list of index names from the Pinecone list_indexes() response.
    Handles various response formats.
    """
    try:
        if hasattr(raw_indexes, "names"):
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
            "pdftotext": {"layout": True},
            "ocr": {"language": "eng"},
        }

    def process_file(self, path: Path) -> Optional[Tuple[Optional[str], Optional[str]]]:
        """
        Process a file based on its MIME type.
        Returns a tuple of (text, mime).
        """
        try:
            mime = magic.from_file(str(path), mime=True)
            if mime not in Config.ACCEPTED_MIME_TYPES:
                logging.info(f"Skipping unsupported MIME type: {mime}")
                return None, mime  # Return None for text, but still mime type
            text = self._dispatch_processing(path, mime)
            return text, mime
        except Exception as e:
            logging.error(f"Error processing {path}: {e}")
            return None, None

    def _dispatch_processing(self, path: Path, mime: str) -> Optional[str]:
        """Dispatch file processing based on MIME type."""
        if mime == "application/pdf":
            return self._process_pdf(path)
        elif mime.startswith("image/"):
            return self._process_image(path)
        elif mime == "application/octet-stream":
            return self._process_octet_stream(path)
        else:
            return self._process_generic(path)

    def _process_octet_stream(self, path: Path) -> Optional[str]:
        """Handle generic binary files by attempting multiple extraction methods."""
        suffix = path.suffix.lower()
        document_exts = [".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".pdf"]
        if suffix in document_exts:
            try:
                if HAS_TEXTRACT:
                    logging.warning(
                        f"Skipping {path} due to textract being unavailable (document extension match)."
                    )
                    return None
                else:
                    print(f"Cannot process {path}: textract not available")
                    return None
            except Exception as e:
                logging.debug(f"Textract failed on binary file {path}: {e}")

        if suffix == ".pdf":
            try:
                return self._process_pdf(path)
            except Exception as e:
                logging.debug(f"PDF processing failed on binary file {path}: {e}")

        image_exts = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"]
        if suffix in image_exts:
            try:
                return self._process_image(path)
            except Exception as e:
                logging.debug(f"Image processing failed on binary file {path}: {e}")

        try:
            if HAS_TEXTRACT:
                logging.warning(
                    f"Skipping {path} due to textract being unavailable (generic fallback)."
                )
                return None
            else:
                print(f"Cannot process {path}: textract not available")
                return None
        except Exception:
            logging.debug(f"All extraction methods failed for binary file {path}")
            return None

    def _process_pdf(self, path: Path) -> Optional[str]:
        """Extract text from PDF using PyMuPDF and OCR fallback, with structured extraction."""
        text = ""
        try:
            structured_chunks = self._process_pdf_with_structure(path)
            for chunk_data in structured_chunks:
                text += chunk_data["text"] + "\n"
            return text
        except Exception as e:
            logging.error(f"PDF processing error ({path}): {e}")
            return None

    def _process_pdf_with_structure(self, path: Path) -> List[Dict]:
        """Extract text from PDF with structural information preserved."""
        structured_chunks = []
        try:
            with fitz.open(str(path)) as doc:
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    blocks = page.get_text("dict")["blocks"]  # type: ignore
                    for block in blocks:
                        if block["type"] == 0:
                            block_text = ""
                            for line in block["lines"]:
                                block_text += (
                                    "".join([span["text"] for span in line["spans"]])
                                    + "\n"
                                )
                            font_size = (
                                block["lines"][0]["spans"][0]["size"]
                                if block["lines"] and block["lines"][0]["spans"]
                                else 12
                            )
                            is_heading = font_size > 14
                            structured_chunks.append(
                                {
                                    "text": block_text.strip(),
                                    "page": page_num + 1,
                                    "is_heading": is_heading,
                                    "font_size": font_size,
                                    "position": block["bbox"],
                                }
                            )
            return structured_chunks
        except Exception as e:
            logging.error(f"Structured PDF processing error ({path}): {e}")
            return []

    def _process_image(self, path: Path) -> Optional[str]:
        """Extract text from an image using Tesseract OCR."""
        try:
            return pytesseract.image_to_string(
                Image.open(path), config="--psm 6 -c preserve_interword_spaces=1"
            )
        except Exception as e:
            logging.error(f"Image processing error ({path}): {e}")
            return None

    def _process_generic(self, path: Path) -> Optional[str]:
        """Extract text using textract; fallback to direct file reading."""
        try:
            if HAS_TEXTRACT:
                logging.warning(
                    f"Skipping {path} due to textract being unavailable (generic processing)."
                )
                return None
            else:
                print(f"Cannot process {path}: textract not available")
                return None
        except Exception as e:
            logging.error(f"Text extraction error ({path}): {e}")
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
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
            self.tokenizers["cl100k_base"] = tiktoken.get_encoding("cl100k_base")
            self.tokenizers["p50k_base"] = tiktoken.get_encoding("p50k_base")
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
        return len(text) // 4

    def get_splitter_for_mimetype(
        self, mime_type: str
    ) -> Any:
        """Dynamically determine text splitter based on MIME type."""
        if mime_type == "application/pdf":
            return RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
            )
        elif mime_type.startswith("text/plain") or mime_type in [
            "text/markdown",
            "text/rtf",
            "application/rtf",
            "text/csv",
            "text/tsv",
        ]:
            return RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
            )
        elif mime_type in [
            "application/x-python",
            "text/x-python",
            "application/javascript",
            "text/javascript",
            "text/css",
        ]:
            return TokenTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
            )
        elif mime_type == "text/html":
            return RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
            )
        else:
            return RecursiveCharacterTextSplitter(
                chunk_size=Config.DEFAULT_CHUNK_SIZE,
                chunk_overlap=Config.DEFAULT_CHUNK_OVERLAP,
                length_function=len,
            )

    def chunk_text(
        self, text: str, metadata: dict, mime_type: str
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Split text into chunks and attach content hash, with detailed analytics.
        Returns a tuple: (chunks, analytics dict)
        """
        if not text:
            return [], {"total_chunks": 0, "total_tokens": 0, "avg_tokens_per_chunk": 0}

        splitter = self.get_splitter_for_mimetype(mime_type)
        chunks = list(splitter.create_documents([text], [metadata]))
        analytics = {
            "total_chunks": len(chunks),
            "total_tokens": 0,
            "token_distribution": [],
            "chunk_sizes": [],
            "mime_type": mime_type,
            "chunk_strategy": str(splitter.__class__.__name__),
        }
        for chunk in chunks:
            chunk.metadata["content_hash"] = hashlib.sha256(
                chunk.page_content.encode()
            ).hexdigest()
            token_count = self.count_tokens(chunk.page_content)
            analytics["token_distribution"].append(token_count)
            analytics["total_tokens"] += token_count
            analytics["chunk_sizes"].append(len(chunk.page_content))
        if chunks:
            analytics["avg_tokens_per_chunk"] = analytics["total_tokens"] / len(chunks)
            analytics["avg_chars_per_chunk"] = sum(analytics["chunk_sizes"]) / len(
                chunks
            )
            analytics["min_tokens"] = (
                min(analytics["token_distribution"])
                if analytics["token_distribution"]
                else 0
            )
            analytics["max_tokens"] = (
                max(analytics["token_distribution"])
                if analytics["token_distribution"]
                else 0
            )
        return chunks, analytics

    def semantic_chunk(self, text: str, metadata: dict, mime_type: str) -> List[Any]:
        """Placeholder for future semantic chunking implementation.
        Uses standard chunking and returns only the chunks.
        """
        logging.info("Semantic chunking placeholder - using standard chunking instead.")
        chunks, _ = self.chunk_text(text, metadata, mime_type)
        return chunks

    def determine_optimal_chunk_parameters(self, query: str) -> Tuple[int, int]:
        """Placeholder for future query-aware chunk parameter determination."""
        logging.info(
            "Query-aware chunking parameter determination placeholder - using defaults."
        )
        return (Config.DEFAULT_CHUNK_SIZE, Config.DEFAULT_CHUNK_OVERLAP)

    def visualize_chunk(self, text: str, model: str = "cl100k_base") -> Dict[str, Any]:
        """Create a visualization of how text would be tokenized and chunked."""
        tokenizer = self.get_tokenizer(model)
        if not tokenizer:
            return {"error": "Tokenizer not available"}
        tokens = tokenizer.encode(text)
        decoded = [
            tokenizer.decode_single_token_bytes(t).decode("utf-8", errors="replace")
            for t in tokens
        ]
        return {
            "original_text": text,
            "token_count": len(tokens),
            "tokens": decoded,
            "character_count": len(text),
            "tokens_per_char": len(tokens) / len(text) if text else 0,
        }


def generate_embeddings(texts: List[str]) -> Optional[List[Any]]:
    """Generate embeddings using the OpenAI client."""
    if oai_client is None:
        return None
    try:
        response = oai_client.embeddings.create(
            input=texts, model=Config.EMBEDDING_MODEL, dimensions=Config.EMBEDDING_DIM
        )
        return [e.embedding for e in response.data]
    except Exception as e:
        logging.error(f"Embedding error: {e}")
        return None
