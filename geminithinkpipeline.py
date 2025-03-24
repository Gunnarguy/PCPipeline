#!/usr/bin/env python3
"""
Refactored version of ultimate_rag_final.py (Version: 6.2)
Improvements:
- Enhanced modularization and clarity
- Added logging for debugging and error handling
- Improved code structure and documentation
"""

import os
import sys
import time
import hashlib
import threading
import queue
import traceback
import logging
from pathlib import Path
from typing import Any, List, Optional, Dict, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

import magic
import textract
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from dotenv import load_dotenv

# External API clients and ML libraries
import pinecone
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Set tokenizer parallelism flag before any other libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(module)s - %(funcName)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


# ======================
# Configuration
# ======================
class Config:
    """Application-wide configuration settings."""
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIM: int = 3072
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 256
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    VALIDATION_THRESHOLD: float = 0.95
    ACCEPTED_MIME_TYPES: set[str] = {
        'application/pdf', 'text/plain',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'image/png', 'image/jpeg', 'text/html'
    }
    MAX_INDEXES: int = 5


# ======================
# System Initialization
# ======================
class SystemInitializer:
    """Initializes API clients, Pinecone, and ML models."""

    @staticmethod
    def initialize() -> Tuple[OpenAI, pinecone.Pinecone, AutoModelForSequenceClassification, AutoTokenizer, torch.device]:
        """Initialize all necessary system components."""
        SystemInitializer._validate_api_keys()
        oai_client = SystemInitializer._initialize_openai_client()
        pc = SystemInitializer._initialize_pinecone_client()
        device = SystemInitializer._get_device()
        reranker, tokenizer = SystemInitializer._load_reranker_model(device)
        return oai_client, pc, reranker, tokenizer, device

    @staticmethod
    def _validate_api_keys() -> None:
        """Ensure required environment variables are set."""
        if not os.getenv("PINECONE_API_KEY") or not os.getenv("OPENAI_API_KEY"):
            logger.error("Missing required API key(s): Check PINECONE_API_KEY and OPENAI_API_KEY")
            sys.exit("Missing required API key(s): Check PINECONE_API_KEY and OPENAI_API_KEY")
        logger.info("API keys validated.")

    @staticmethod
    def _initialize_openai_client() -> OpenAI:
        """Initialize the OpenAI client."""
        client = OpenAI()
        logger.info("OpenAI client initialized.")
        return client

    @staticmethod
    def _initialize_pinecone_client() -> pinecone.Pinecone:
        """Initialize the Pinecone client."""
        client = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        logger.info("Pinecone client initialized.")
        return client

    @staticmethod
    def _get_device() -> torch.device:
        """Determine and return the appropriate device (CUDA or CPU)."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        return device

    @staticmethod
    def _load_reranker_model(device: torch.device) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """Load the reranker model and tokenizer."""
        try:
            reranker = AutoModelForSequenceClassification.from_pretrained(Config.RERANKER_MODEL).to(device)
            tokenizer = AutoTokenizer.from_pretrained(Config.RERANKER_MODEL)
            logger.info(f"Reranker model '{Config.RERANKER_MODEL}' loaded onto {device}.")
            return reranker, tokenizer
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            sys.exit(f"Model loading failed: {e}")


# ======================
# Helper Functions
# ======================
class PineconeHelper:
    """Helper functions for interacting with Pinecone."""

    @staticmethod
    def wait_for_index_ready(pc_client: pinecone.Pinecone, index_name: str, timeout: int = 60, poll_interval: int = 1) -> bool:
        """Wait until the specified index is ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                desc = pc_client.describe_index(index_name)
                if desc.get("status", {}).get("state") == "Ready":
                    logger.info(f"Index '{index_name}' is ready.")
                    return True
            except pinecone.core.client.exceptions.NotFoundException:
                pass
            time.sleep(poll_interval)
        logger.warning(f"Timeout waiting for index '{index_name}'.")
        return False

    @staticmethod
    def extract_index_names(raw_indexes: Any) -> List[str]:
        """
        Extract a list of index names from the Pinecone list_indexes() response.
        Handles various response formats.
        """
        index_names: List[str] =
        try:
            if hasattr(raw_indexes, 'names') and callable(raw_indexes.names):
                index_names.extend(raw_indexes.names())
            elif isinstance(raw_indexes, dict) and "indexes" in raw_indexes:
                descs = raw_indexes["indexes"]
                if descs and isinstance(descs[0], dict) and "name" in descs[0]:
                    index_names.extend([d["name"] for d in descs])
                elif isinstance(descs, list) and all(isinstance(item, str) for item in descs):
                    index_names.extend(descs)
            elif isinstance(raw_indexes, list):
                for item in raw_indexes:
                    if isinstance(item, dict) and "name" in item:
                        index_names.append(item["name"])
                    elif isinstance(item, str):
                        index_names.append(item)
            elif isinstance(raw_indexes, str):
                index_names.append(raw_indexes)
        except Exception as e:
            logger.error(f"Error extracting index names: {e}")
        logger.debug(f"Extracted index names: {index_names}")
        return list(set(index_names))


# ======================
# File Processor
# ======================
class FileProcessor:
    """Class to handle file processing based on MIME type."""

    def __init__(self) -> None:
        """Initializes the FileProcessor with textract configuration."""
        self.textract_config = {
            'pdftotext': {'layout': True},
            'ocr': {'language': 'eng'}
        }
        logger.info("FileProcessor initialized.")

    def process_file(self, path: Path) -> Optional[str]:
        """Process a file based on its MIME type."""
        try:
            mime = magic.from_file(str(path), mime=True)
            logger.debug(f"Detected MIME type for {path}: {mime}")
            if mime not in Config.ACCEPTED_MIME_TYPES:
                logger.info(f"Skipping unsupported MIME type '{mime}' for file: {path}")
                return None
            return self._dispatch_processing(path, mime)
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            return None
        except Exception as e:
            logger.error(f"Error processing file {path}: {e}")
            return None

    def _dispatch_processing(self, path: Path, mime: str) -> Optional[str]:
        """Dispatch file processing based on MIME type."""
        if mime == 'application/pdf':
            return self._process_pdf(path)
        elif mime.startswith('image/'):
            return self._process_image(path)
        else:
            return self._process_generic(path)

    def _process_pdf(self, path: Path) -> Optional[str]:
        """Extract text from PDF using PyMuPDF and OCR fallback."""
        text = ""
        try:
            with fitz.open(str(path)) as doc:
                for page in doc:
                    page_text = page.get_text("text")
                    text += page_text
                    if len(page_text.strip()) < 100:
                        logger.info(f"Applying OCR fallback for page {page.number + 1} of {path}")
                        pix = page.get_pixmap(dpi=300)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        text += "\n" + pytesseract.image_to_string(img)
            logger.info(f"Successfully processed PDF: {path}")
            return text
        except Exception as e:
            logger.error(f"PDF processing error for {path}: {e}")
            return None

    def _process_image(self, path: Path) -> Optional[str]:
        """Extract text from an image using Tesseract OCR."""
        try:
            text = pytesseract.image_to_string(
                Image.open(path),
                config='--psm 6 -c preserve_interword_spaces=1'
            )
            logger.info(f"Successfully processed image: {path}")
            return text
        except Exception as e:
            logger.error(f"Image processing error for {path}: {e}")
            return None

    def _process_generic(self, path: Path) -> Optional[str]:
        """Extract text using textract; fallback to direct file reading."""
        try:
            text = textract.process(str(path), **self.textract_config).decode('utf-8')
            logger.info(f"Successfully processed generic file: {path}")
            return text
        except Exception as e:
            logger.warning(f"Text extraction error using textract for {path}: {e}. Attempting direct file reading.")
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    logger.info(f"Successfully read text directly from file: {path}")
                    return text
            except Exception:
                logger.error(f"Failed to extract text from {path} using both textract and direct reading.")
                return None


# ======================
# Text Processor
# ======================
class TextProcessor:
    """Class for text splitting and chunking."""

    def __init__(self) -> None:
        """Initializes the TextProcessor with a RecursiveCharacterTextSplitter."""
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True  # Helpful for context tracking if needed later
        )
        logger.info("TextProcessor initialized.")

    def chunk_text(self, text: str, metadata: dict) -> List[Any]:
        """Split text into chunks and attach metadata, including a content hash."""
        if not text.strip():
            logger.warning("Attempted to chunk empty text.")
            return
        chunks = self.splitter.create_documents([text], [metadata])
        for chunk in chunks:
            chunk.metadata['content_hash'] = hashlib.sha256(chunk.page_content.encode()).hexdigest()
        logger.debug(f"Chunked text into {len(chunks)} parts.")
        return chunks


# ======================
# Embeddings Generator
# ======================
class EmbeddingsGenerator:
    """Generates embeddings for text using the OpenAI API."""

    def __init__(self, client: OpenAI) -> None:
        """Initializes the EmbeddingsGenerator with an OpenAI client."""
        self.client = client
        logger.info("EmbeddingsGenerator initialized.")

    def generate_embeddings(self, texts: List[str]) -> Optional[List[Any]]:
        """Generate embeddings using the OpenAI client."""
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=Config.EMBEDDING_MODEL,
                dimensions=Config.EMBEDDING_DIM
            )
            embeddings = [e.embedding for e in response.data]
            logger.debug(f"Generated {len(embeddings)} embeddings.")
            return embeddings
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return None


# ======================
# Reranker
# ======================
class Reranker:
    """Reranks search results using a cross-encoder model."""

    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, device: torch.device) -> None:
        """Initializes the Reranker with a model, tokenizer, and device."""
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        logger.info(f"Reranker initialized with model '{Config.RERANKER_MODEL}' on {device}.")

    def rerank(self, query: str, hits: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float]]:
        """Reranks search hits based on the query."""
        if not hits:
            return
        pairs = [(query, hit.metadata.get('text', '')) for hit in hits]
        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        if logits.shape[1] >= 2:
            scores = torch.softmax(logits, dim=1)[:, 1].cpu().detach().numpy()
        else:
            scores = torch.sigmoid(logits).squeeze(-1).cpu().detach().numpy()
        reranked_results = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
        logger.debug(f"Reranked {len(hits)} search results.")
        return reranked_results


# ======================
# Guided RAG System GUI
# ======================
class GuidedRAGInterface(tk.Tk):
    """Tkinter-based GUI for the Universal RAG System."""

    def __init__(self, oai_client: OpenAI, pc_client: pinecone.Pinecone, reranker_model: AutoModelForSequenceClassification, reranker_tokenizer: AutoTokenizer, device: torch.device) -> None:
        super().__init__()
        self.title("Universal RAG System")
        self.geometry("1200x800")
        self.queue = queue.Queue()

        # API Clients and Models
        self.oai_client = oai_client
        self.pc_client = pc_client
        self.reranker_model = reranker_model
        self.reranker_tokenizer = reranker_tokenizer
        self.device = device

        # State variables
        self.selected_index = tk.StringVar()
        self.selected_namespace = tk.StringVar()
        self.setup_complete = False
        self.current_step = "welcome"
        self.pc_index: Optional[pinecone.Index] = None

        # Initialize processors
        self.file_processor = FileProcessor()
        self.text_processor = TextProcessor()
        self.embeddings_generator = EmbeddingsGenerator(self.oai_client)
        self.reranker = Reranker(self.reranker_model, self.reranker_tokenizer, self.device)

        # Build UI
        self._create_widgets()
        self.after(100, self._process_queue)
        self.start_guided_setup()
        logger.info("GUI initialized.")

    def _create_widgets(self) -> None:
        """Create and configure all GUI widgets."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Main frame
        self.main_frame = ttk.Frame(self)
        self.main_frame.grid(row=0, column=0, rowspan=3, sticky="nsew", padx=10, pady=10)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(2, weight=1)

        # Setup (wizard) frame
        self.setup_frame = ttk.Frame(self.main_frame)
        self.setup_frame.grid(row=0, column=0, sticky="nsew")
        self.setup_frame.columnconfigure(0, weight=1)

        self.setup_title = ttk.Label(self.setup_frame, text="Welcome to Universal RAG System", font=("Arial", 14, "bold"))
        self.setup_title.grid(row=0, column=0, pady=20)

        self.setup_instructions = ttk.Label(self.setup_frame, text="", wraplength=800, justify="center")
        self.setup_instructions.grid(row=1, column=0, pady=10)

        self.options_frame = ttk.Frame(self.setup_frame)
        self.options_frame.grid(row=2, column=0, pady=20)

        self.button_frame = ttk.Frame(self.setup_frame)
        self.button_frame.grid(row=3, column=0, pady=20)

        self.back_btn = ttk.Button(self.button_frame, text="Back", command=self.handle_back)
        self.back_btn.grid(row=0, column=0, padx=10)

        self.next_btn = ttk.Button(self.button_frame, text="Next")
        self.next_btn.grid(row=0, column=1, padx=10)

        # Main application frame (post-setup)
        self.app_frame = ttk.Frame(self.main_frame)
        self.app_frame.grid(row=0, column=0, rowspan=3, sticky="nsew")
        self.app_frame.grid_remove()

        self._create_app_controls()
        logger.debug("GUI widgets created.")

    def _create_app_controls(self) -> None:
        """Create controls for document selection, indexing, and search."""
        # Control Frame
        control_frame = ttk.Frame(self.app_frame)
        control_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        control_frame.columnconfigure(tuple(range(9)), weight=1)

        self.dir_btn = ttk.Button(control_frame, text="Select Documents", command=self._select_directory)
        self.dir_btn.grid(row=0, column=0, padx=5, pady=2)

        ttk.Label(control_frame, text="Index:").grid(row=0, column=1, padx=5, pady=2)
        self.index_combo = ttk.Combobox(control_frame, textvariable=self.selected_index, state="readonly", width=20)
        self.index_combo.grid(row=0, column=2, padx=5, pady=2)
        self.index_combo.bind("<<ComboboxSelected>>", self.on_index_change)

        self.new_index_btn = ttk.Button(control_frame, text="New Index", command=self.create_new_index)
        self.new_index_btn.grid(row=0, column=3, padx=5, pady=2)

        self.refresh_indexes_btn = ttk.Button(control_frame, text="Refresh Indexes", command=self.refresh_index_list)
        self.refresh_indexes_btn.grid(row=0, column=4, padx=5, pady=2)

        ttk.Label(control_frame, text="Namespace:").grid(row=0, column=5, padx=5, pady=2)
        self.namespace_combo = ttk.Combobox(control_frame, textvariable=self.selected_namespace, state="readonly", width=20)
        self.namespace_combo.grid(row=0, column=6, padx=5, pady=2)

        self.new_namespace_btn = ttk.Button(control_frame, text="New Namespace", command=self.create_new_namespace)
        self.new_namespace_btn.grid(row=0, column=7, padx=5, pady=2)

        self.refresh_namespaces_btn = ttk.Button(control_frame, text="Refresh Namespaces", command=self.refresh_namespace_list)
        self.refresh_namespaces_btn.grid(row=0, column=8, padx=5, pady=2)

        self.progress = ttk.Progressbar(control_frame, orient="horizontal", mode="determinate")
        self.progress.grid(row=1, column=0, columnspan=4, sticky="ew", padx=5, pady=2)

        self.status = ttk.Label(control_frame, text="Ready")
        self.status.grid(row=1, column=4, columnspan=5, padx=5, pady=2)

        # Query Frame
        query_frame = ttk.Frame(self.app_frame)
        query_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        query_frame.columnconfigure(0, weight=1)

        self.query_entry = ttk.Entry(query_frame)
        self.query_entry.grid(row=0, column=0, sticky="ew", padx=5)
        self.search_btn = ttk.Button(query_frame, text="Search", command=self._execute_search)
        self.search_btn.grid(row=0, column=1, padx=5)

        # Results Frame
        results_frame = ttk.Frame(self.app_frame)
        results_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)

        self.results = tk.Text(results_frame, wrap=tk.WORD)
        self.results.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.results.config(yscrollcommand=scrollbar.set)
        logger.debug("Application controls created.")

    def _toggle_ui_state(self, enabled: bool) -> None:
        """Enable or disable UI elements."""
        state = "normal" if enabled else "disabled"
        widgets = [self.dir_btn, self.search_btn, self.query_entry,
                   self.index_combo, self.namespace_combo, self.refresh_indexes_btn,
                   self.refresh_namespaces_btn, self.new_index_btn, self.new_namespace_btn]
        for widget in widgets:
            widget.config(state=state)
        self.status.config(text="Ready" if enabled else "Processing...")
        logger.debug(f"UI state toggled to {'enabled' if enabled else 'disabled'}.")

    def _process_queue(self) -> None:
        """Process queued UI updates from background threads."""
        while not self.queue.empty():
            try:
                task = self.queue.get_nowait()
                task()
            except queue.Empty:
                break
            except Exception as e:
                self.status.config(text=f"Queue error: {e}")
                logger.error(f"Error processing queue: {traceback.format_exc()}")
        self.after(100, self._process_queue)

    def _queue_ui_task(self, task) -> None:
        """Helper to add a task to the UI update queue."""
        self.queue.put(task)
        logger.debug("Task added to UI queue.")

    # -------------------------
    # Index / Namespace Handling
    # -------------------------
    def create_new_index(self) -> None:
        """Prompt user to create a new index and create it if valid."""
        new_index = simpledialog.askstring("New Index", "Enter a name for the new index:")
        if not new_index:
            messagebox.showwarning("Input Required", "Please enter an index name.")
            return

        if not new_index.isalnum():
            messagebox.showerror("Invalid Index Name", "Index name must be alphanumeric.")
            return

        if new_index.isdigit():
            new_index = f"index-{new_index}"

        try:
            raw_indexes = self.pc_client.list_indexes()
            existing_indexes = PineconeHelper.extract_index_names(raw_indexes)
            if new_index in existing_indexes:
                messagebox.showinfo("Index Exists", f"Index '{new_index}' already exists. Selecting it.")
                self.selected_index.set(new_index)
                self._update_pc_index(new_index)
                return

            if len(existing_indexes) >= Config.MAX_INDEXES:
                messagebox.showerror("Index Creation Error", f"Cannot create new index. Max of {Config.MAX_INDEXES} reached.")
                return

            self.pc_client.create_index(
                name=new_index,
                dimension=Config.EMBEDDING_DIM,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
            )
            if PineconeHelper.wait_for_index_ready(self.pc_client, new_index):
                messagebox.showinfo("Index Created", f"Index '{new_index}' is ready.")
                self.refresh_index_list()
                self.selected_index.set(new_index)
                self._update_pc_index(new_index)
            else:
                messagebox.showerror("Timeout", f"Index '{new_index}' was not ready in time.")
        except Exception as e:
            messagebox.showerror("Index Creation Error", str(e))
        logger.info(f"Attempted to create index: {new_index}")

    def create_new_namespace(self) -> None:
        """Prompt user to create a new namespace."""
        new_namespace = simpledialog.askstring("New Namespace", "Enter a name for the new namespace:")
        if new_namespace:
            current_namespaces = list(self.namespace_combo['values'])
            if new_namespace not in current_namespaces:
                current_namespaces.append(new_namespace)
                self.namespace_combo['values'] = sorted(current_namespaces)
            self.selected_namespace.set(new_namespace)
            messagebox.showinfo("Namespace Set", f"Namespace set to '{new_namespace}'.")
            logger.info(f"Namespace set to: {new_namespace}")

    def refresh_index_list(self) -> None:
        """Refresh the list of indexes from Pinecone."""
        try:
            raw_indexes = self.pc_client.list_indexes()
            indexes = PineconeHelper.extract_index_names(raw_indexes)
            if not indexes:
                messagebox.showinfo("No Indexes", "No indexes found. Please create one.")
                self.index_combo['values'] =
                self.selected_index.set("")
                self.pc_index = None
                return

            self.index_combo['values'] = sorted(indexes)
            if self.selected_index.get() not in indexes and indexes:
                self.selected_index.set(indexes[0])
                self._update_pc_index(indexes[0])
            self.refresh_namespace_list()
        except Exception as e:
            messagebox.showerror("Index Refresh Error", f"Error refreshing indexes: {e}")
        logger.info("Index list refreshed.")

    def _update_pc_index(self, index_name: str) -> None:
        """Update the global Pinecone index object."""
        try:
            self.pc_index = self.pc_client.Index(index_name)
            logger.info(f"Pinecone index updated to: {index_name}")
        except Exception as e:
            messagebox.showerror("Index Update Error", str(e))

    def refresh_namespace_list(self) -> None:
        """Refresh the namespace list from the current Pinecone index."""
        try:
            if not self.pc_index:
                self.namespace_combo['values'] =
                self.selected_namespace.set("")
                return

            stats = self.pc_index.describe_index_stats()
            namespaces =
            if isinstance(stats, dict) and "namespaces" in stats:
                namespaces.extend(stats["namespaces"].keys())
            elif hasattr(stats, 'namespaces') and isinstance(stats.namespaces, dict):
                namespaces.extend(stats.namespaces.keys())

            self.namespace_combo['values'] = sorted(namespaces)
            if self.selected_namespace.get() not in namespaces and namespaces:
                self.selected_namespace.set(namespaces[0])
        except Exception as e:
            messagebox.showerror("Namespace Refresh Error", f"Error refreshing namespaces: {e}")
        logger.info("Namespace list refreshed.")

    def on_index_change(self, event: tk.Event) -> None:
        """Handle index selection change."""
        index_name = self.selected_index.get()
        self._update_pc_index(index_name)
        self.refresh_namespace_list()
        logger.info(f"Index changed to: {index_name}")

    # -------------------------
    # Document Processing
    # -------------------------
    def _select_directory(self) -> None:
        """Open a dialog for selecting a directory and process its files."""
        path = filedialog.askdirectory()
        if path:
            logger.info(f"Directory selected for processing: {path}")
            self._toggle_ui_state(False)
            threading.Thread(target=self._process_files_in_thread, args=(path,), daemon=True).start()

    def _process_files_in_thread(self, path: str) -> None:
        """Wrapper for _process_files to run in a separate thread."""
        try:
            self._process_files(path)
        except Exception as e:
            logger.error(f"Error during file processing thread: {traceback.format_exc()}")
            self._queue_ui_task(lambda: messagebox.showerror("Processing Error", str(e)))
        finally:
            self._queue_ui_task(lambda: self._toggle_ui_state(True))

    def _process_files(self, path: str) -> None:
        """Process all files in the selected directory and upsert embeddings."""
        try:
            file_paths = [Path(root) / f for root, _, files in os.walk(path) for f in files]
            total_files = len(file_paths)
            processed_files = 0
            all_vectors =
            namespace = self.selected_namespace.get() or None
            logger.info(f"Processing {total_files} files from: {path}")

            for idx, file_path in enumerate(file_paths):
                logger.info(f"Processing file: {file_path} ({idx + 1}/{total_files})")
                text = self.file_processor.process_file(file_path)
                if text:
                    chunks = self.text_processor.chunk_text(text, {"source": str(file_path)})
                    texts_to_embed = [chunk.page_content for chunk in chunks]
                    embeddings = self.embeddings_generator.generate_embeddings(texts_to_embed)
                    if embeddings:
                        vectors =
                        for chunk, emb in zip(chunks, embeddings):
                            vectors.append({
                                "id": f"{chunk.metadata['content_hash']}_{idx}",
                                "values": emb,
                                "metadata": {
                                    "text": chunk.page_content,
                                    "source": str(file_path),
                                    "hash": chunk.metadata['content_hash']
                                }
                            })
                        all_vectors.extend(vectors)
                        logger.debug(f"Generated {len(vectors)} embeddings for {file_path}")
                    else:
                        logger.warning(f"No embeddings generated for {file_path}")
                else:
                    logger.warning(f"No text extracted from {file_path}")

                processed_files += 1
                progress_value = (processed_files / total_files) * 100
                self._queue_ui_task(lambda v=progress_value: self.progress.configure(value=v))
                self._queue_ui_task(lambda p=processed_files, t=total_files: self.status.configure(text=f"Processed {p}/{t} files"))

            if all_vectors:
                logger.info(f"Upserting {len(all_vectors)} vectors to Pinecone (namespace: {namespace}).")
                for i in range(0, len(all_vectors), 100):
                    try:
                        self.pc_index.upsert(vectors=all_vectors[i:i + 100], namespace=namespace)
                        logger.debug(f"Upserted vectors {i} to {i + len(all_vectors[i:i + 100]) - 1}.")
                    except Exception as e:
                        logger.error(f"Error during upsert: {e}")
                        self._queue_ui_task(lambda: messagebox.showerror("Pinecone Upsert Error", str(e)))
                        return

                self._queue_ui_task(lambda: messagebox.showinfo("Processing Complete", f"Indexed {len(all_vectors)} chunks from {processed_files} files"))
            else:
                self._queue_ui_task(lambda: messagebox.showinfo("Processing Complete", "No usable content found in the selected directory."))

        except Exception as e:
            logger.error(f"Error processing files in {path}: {traceback.format_exc()}")
            self._queue_ui_task(lambda: messagebox.showerror("Processing Error", str(e)))

    # -------------------------
    # Searching
    # -------------------------
    def _execute_search(self) -> None:
        """Initiate a search query from user input."""
        query = self.query_entry.get().strip()
        if not query:
            messagebox.showwarning("Empty Query", "Please enter a search term")
            return
        logger.info(f"Executing search for query: '{query}'")
        self._toggle_ui_state(False)
        threading.Thread(target=self._perform_search_in_thread, args=(query,), daemon=True).start()

    def _perform_search_in_thread(self, query: str) -> None:
        """Wrapper for _perform_search to run in a separate thread."""
        try:
            self._perform_search(query)
        except Exception as e:
            logger.error(f"Error during search thread: {traceback.format_exc()}")
            self._queue_ui_task(lambda: messagebox.showerror("Search Error", str(e)))
        finally:
            self._queue_ui_task(lambda: self._toggle_ui_state(True))

    def _perform_search(self, query: str) -> None:
        """Perform the search and rerank results using the reranker model."""
        try:
            namespace = self.selected_namespace.get() or None
            embedding_list = self.embeddings_generator.generate_embeddings([query])
            if not embedding_list:
                raise ValueError("Embedding generation returned None.")
            query_embedding = embedding_list[0]

            results = self.pc_index.query(
                vector=query_embedding,
                top_k=50,
                include_metadata=True,
                namespace=namespace
            )
            hits = results.matches if results and results.matches else
            if not hits:
                self._queue_ui_task(lambda: self.results.delete(1.0, tk.END))
                self._queue_ui_task(lambda: self.results.insert(tk.END, "No matching results found."))
                return

            reranked_results = self.reranker.rerank(query, hits)

            context = "\n\n".join([
                f"Source: {res.metadata.get('source', 'Unknown')}\n{res.metadata.get('text', 'No content available')}"
                for res, _ in reranked_results[:10]
            ])

            answer = self.oai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"Answer the question using ONLY the following sources:\n{context}"},
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                max_tokens=1000
            ).choices[0].message.content

            result_text = f"Query: {query}\n\nAnswer: {answer}\n\nTop Sources:\n"
            for idx, (match, score) in enumerate(reranked_results[:5]):  # Display top 5 sources
                result_text += f"{idx + 1}: \"{match.metadata.get('source', 'Unknown')}\" (Score: {score:.3f})\n"

            self._queue_ui_task(lambda: self.results.delete(1.0, tk.END))
            self._queue_ui_task(lambda rt=result_text: self.results.insert(tk.END, rt))
            logger.info(f"Search completed for query: '{query}'.")

        except Exception as e:
            error_msg = str(e)
            trace = traceback.format_exc()
            self._queue_ui_task(lambda em=error_msg, tb=trace: messagebox.showerror(
                "Search Error",
                f"Error during search:\n{em}\n\nTechnical details:\n{tb}"
            ))

    # -------------------------
    # Wizard Flow
    # -------------------------
    def start_guided_setup(self) -> None:
        """Begin the setup wizard for the RAG system."""
        self.current_step = "welcome"
        self.setup_complete = False

        self.app_frame.grid_remove()
        self.setup_frame.grid()

        self.setup_title.config(text="Welcome to Universal RAG System")
        self.setup_instructions.config(
            text="This wizard will guide you through setting up your RAG system.\n"
                 "First, you'll need to select or create an index."
        )

        for widget in self.options_frame.winfo_children():
            widget.grid_remove()

        self.back_btn.config(state="disabled")
        self.next_btn.config(text="Start Setup", command=self.show_index_selection)
        logger.info("Guided setup started.")

    def show_index_selection(self) -> None:
        """Display options for index selection."""
        self.current_step = "index_selection"
        self.setup_title.config(text="Index Selection")
        self.setup_instructions.config(text="Do you want to create a new index or select an existing one?")

        for widget in self.options_frame.winfo_children():
            widget.grid_remove()

        self.setup_choice = tk.StringVar()
        self.radio_frame = ttk.Frame(self.options_frame)
        self.radio_frame.grid(row=0, column=0)

        create_radio = ttk.Radiobutton(self.radio_frame, text="Create new index", value="create", variable=self.setup_choice)
        create_radio.grid(row=0, column=0, pady=5, sticky="w")

        select_radio = ttk.Radiobutton(self.radio_frame, text="Select existing index", value="select", variable=self.setup_choice)
        select_radio.grid(row=1, column=0, pady=5, sticky="w")

        self.back_btn.config(state="normal")
        self.next_btn.config(text="Next", command=self.handle_next)
        logger.debug("Showing index selection options.")

    def handle_back(self) -> None:
        """Handle the Back button action based on current step."""
        logger.debug(f"Back button pressed from step: {self.current_step}")
        if self.current_step in ["select_index", "create_index", "setup_complete", "select_namespace", "create_namespace"]:
            self.show_index_selection()

    def handle_next(self) -> None:
        """Handle the Next button action based on current step."""
        logger.debug(f"Next button pressed from step: {self.current_step}")
        if self.current_step == "index_selection":
            choice = self.setup_choice.get()
            if choice == "create":
                self.show_create_index()
            elif choice == "select":
                self.show_select_index()
            else:
                messagebox.showwarning("Selection Required", "Please select an option to continue.")
        elif self.current_step == "select_index":
            selected_indices = self.setup_listbox.curselection()
            if not selected_indices:
                messagebox.showwarning("Selection Required", "Please select an index.")
                return
            index_name = self.setup_listbox.get(selected_indices[0])
            self.selected_index.set(index_name)
            self._update_pc_index(index_name)
            self.show_namespace_selection_choice()
        elif self.current_step == "create_index":
            index_name = self.setup_entry.get().strip()
            if not index_name:
                messagebox.showwarning("Input Required", "Please enter an index name.")
                return
            self.create_index_and_continue(index_name)
        elif self.current_step == "namespace_selection_choice":
            choice = self.setup_choice.get()
            if choice == "create":
                self.show_create_namespace()
            elif choice == "select":
                self.show_select_namespace()
            else:
                self.selected_namespace.set("")  # Default to no namespace
                self.show_setup_complete()
        elif self.current_step == "select_namespace":
            selected_namespaces = self.setup_listbox.curselection()
            if selected_namespaces:
                ns_display = self.setup_listbox.get(selected_namespaces[0])
                namespace = ns_display if ns_display != "(default)" else ""
                self.selected_namespace.set(namespace)
            else:
                self.selected_namespace.set("")  # Default to no namespace
            self.show_setup_complete()
        elif self.current_step == "create_namespace":
            namespace = self.setup_entry.get().strip()
            if namespace:
                self.selected_namespace.set(namespace)
            else:
                self.selected_namespace.set("") # Default to no namespace
            self.show_setup_complete()

    def show_select_index(self) -> None:
        """Display a list for selecting an existing index."""
        self.current_step = "select_index"
        self.setup_title.config(text="Select Existing Index")
        self.setup_instructions.config(text="Select an index from the list below.")

        for widget in self.options_frame.winfo_children():
            widget.grid_remove()

        self.setup_listbox = tk.Listbox(self.options_frame, width=40, height=10)
        self.setup_listbox.grid(row=0, column=0, pady=10)
        self.setup_scrollbar = ttk.Scrollbar(self.options_frame, orient="vertical", command=self.setup_listbox.yview)
        self.setup_listbox.config(yscrollcommand=self.setup_scrollbar.set)
        self.setup_scrollbar.grid(row=0, column=1, sticky="ns")

        try:
            raw_indexes = self.pc_client.list_indexes()
            indexes = PineconeHelper.extract_index_names(raw_indexes)
            if not indexes:
                messagebox.showinfo("No Indexes", "No existing indexes found. Please create one.")
                self.show_create_index()
                return
            for idx_name in sorted(indexes):
                self.setup_listbox.insert(tk.END, idx_name)
            if self.setup_listbox.size() > 0:
                self.setup_listbox.selection_set(0)
        except Exception as e:
            messagebox.showerror("Index List Error", f"Error listing indexes: {e}")
            self.show_index_selection()
        logger.debug("Showing select index UI.")

    def show_create_index(self) -> None:
        """Display UI for creating a new index."""
        self.current_step = "create_index"
        self.setup_title.config(text="Create New Index")
        self.setup_instructions.config(text="Enter a name for your new index.")

        for widget in self.options_frame.winfo_children():
            widget.grid_remove()

        self.setup_entry = ttk.Entry(self.options_frame, width=40)
        self.setup_entry.grid(row=0, column=0, pady=10)
        self.next_btn.config(text="Create", command=lambda: self.create_index_and_continue(self.setup_entry.get()))
        logger.debug("Showing create index UI.")

    def create_index_and_continue(self, index_name: str) -> None:
        """Create a new index and proceed to the namespace selection."""
        try:
            if not index_name.isalnum():
                messagebox.showerror("Invalid Index Name", "Index name must be alphanumeric.")
                return
            if index_name.isdigit():
                index_name = f"index-{index_name}"

            raw_indexes = self.pc_client.list_indexes()
            existing_indexes = PineconeHelper.extract_index_names(raw_indexes)
            if index_name in existing_indexes:
                messagebox.showinfo("Index Exists", f"Index '{index_name}' already exists. Selecting it.")
                self.selected_index.set(index_name)
                self._update_pc_index(index_name)
                self.show_namespace_selection_choice()
                return

            if len(existing_indexes) >= Config.MAX_INDEXES:
                messagebox.showerror("Index Creation Error", f"Cannot create new index. Maximum of {Config.MAX_INDEXES} indexes reached.")
                return

            self.pc_client.create_index(
                name=index_name,
                dimension=Config.EMBEDDING_DIM,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
            )
            if PineconeHelper.wait_for_index_ready(self.pc_client, index_name):
                messagebox.showinfo("Index Created", f"Index '{index_name}' is ready for use.")
                self.selected_index.set(index_name)
                self._update_pc_index(index_name)
                self.show_namespace_selection_choice()
            else:
                messagebox.showerror("Timeout Error", f"Index '{index_name}' did not become ready in time.")
        except Exception as e:
            if "409" in str(e) or "ALREADY_EXISTS" in str(e):
                messagebox.showinfo("Index Exists", f"Index '{index_name}' already exists. Selecting it.")
                self.selected_index.set(index_name)
                self._update_pc_index(index_name)
                self.show_namespace_selection_choice()
            else:
                messagebox.showerror("Index Creation Error", str(e))
        logger.info(f"Attempted to create index: {index_name}")

    def show_namespace_selection_choice(self) -> None:
        """Display options for namespace selection."""
        self.current_step = "namespace_selection_choice"
        self.setup_title.config(text="Namespace Selection")
        self.setup_instructions.config(text="Do you want to select an existing namespace, create a new one, or use the default (no namespace)?")

        for widget in self.options_frame.winfo_children():
            widget.grid_remove()

        self.setup_choice = tk.StringVar(value="default") # Default to no namespace
        self.radio_frame = ttk.Frame(self.options_frame)
        self.radio_frame.grid(row=0, column=0)

        default_radio = ttk.Radiobutton(self.radio_frame, text="Use default (no namespace)", value="default", variable=self.setup_choice)
        default_radio.grid(row=0, column=0, pady=5, sticky="w")

        create_radio = ttk.Radiobutton(self.radio_frame, text="Create new namespace", value="create", variable=self.setup_choice)
        create_radio.grid(row=1, column=0, pady=5, sticky="w")

        select_radio = ttk.Radiobutton(self.radio_frame, text="Select existing namespace", value="select", variable=self.setup_choice)
        select_radio.grid(row=2, column=0, pady=5, sticky="w")

        self.back_btn.config(state="normal")
        self.next_btn.config(text="Next", command=self.handle_next)
        logger.debug("Showing namespace selection options.")

    def show_select_namespace(self) -> None:
        """Display a list for selecting an existing namespace."""
        self.current_step = "select_namespace"
        self.setup_title.config(text="Select Existing Namespace")
        self.setup_instructions.config(text="Select a namespace from the list below (or choose default in the previous step).")

        for widget in self.options_frame.winfo_children():
            widget.grid_remove()

        self.setup_listbox = tk.Listbox(self.options_frame, width=40, height=10)
        self.setup_listbox.grid(row=0, column=0, pady=10)
        self.setup_scrollbar = ttk.Scrollbar(self.options_frame, orient="vertical", command=self.setup_listbox.yview)
        self.setup_listbox.config(yscrollcommand=self.setup_scrollbar.set)
        self.setup_scrollbar.grid(row=0, column=1, sticky="ns")

        try:
            if self.pc_index:
                stats = self.pc_index.describe_index_stats()
                namespaces =
                if isinstance(stats, dict) and "namespaces" in stats:
                    namespaces.extend(stats["namespaces"].keys())
                elif hasattr(stats, 'namespaces') and isinstance(stats.namespaces, dict):
                    namespaces.extend(stats.namespaces.keys())

                for ns in sorted(namespaces):
                    self.setup_listbox.insert(tk.END, ns)
                if self.setup_listbox.size() > 0:
                    self.setup_listbox.selection_set(0)
            else:
                messagebox.showinfo("No Index Selected", "Please select an index first.")
                self.show_index_selection()
                return
        except Exception as e:
            messagebox.showerror("Namespace List Error", f"Error listing namespaces: {e}")
            self.show_namespace_selection_choice()
        logger.debug("Showing select namespace UI.")

    def show_create_namespace(self) -> None:
        """Display UI for creating a new namespace."""
        self.current_step = "create_namespace"
        self.setup_title.config(text="Create New Namespace")
        self.setup_instructions.config(text="Enter a name for your new namespace (or leave blank for default).")

        for widget in self.options_frame.winfo_children():
            widget.grid_remove()

        self.setup_entry = ttk.Entry(self.options_frame, width=40)
        self.setup_entry.grid(row=0, column=0, pady=10)
        self.next_btn.config(text="Finish Setup", command=self.handle_next)
        logger.debug("Showing create namespace UI.")

    def show_setup_complete(self) -> None:
        """Complete the setup and switch to the main application UI."""
        self.current_step = "setup_complete"
        self.setup_frame.grid_remove()
        self.app_frame.grid()
        self.setup_complete = True
        self.refresh_index_list()
        self.refresh_namespace_list()
        logger.info("Guided setup complete. Showing main application UI.")


# ======================
# Main Execution
# ======================
def main() -> None:
    """Main entry point for the application."""
    # Check for required packages
    try:
        import magic  # noqa: F401
        import textract  # noqa: F401
    except ImportError as e:
        print(f"Missing required package: {e.name}. Please install it.")
        if e.name == 'magic':
            print("  brew install libmagic")
            print("  pip install python-magic")
        elif e.name == 'textract':
            print("  brew install tesseract poppler")
            print("  pip install textract")
        sys.exit(1)

    try:
        oai_client, pc_client, reranker_model, reranker_tokenizer, device = SystemInitializer.initialize()
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(f"Initialization failed: {e}")

    app = GuidedRAGInterface(oai_client, pc_client, reranker_model, reranker_tokenizer, device)
    app.mainloop()
    logger.info("Application finished.")


if __name__ == "__main__":
    main()
