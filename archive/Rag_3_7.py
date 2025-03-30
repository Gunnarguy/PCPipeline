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
from typing import Any, List, Optional

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
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Load environment variables
load_dotenv()


# ======================
# Configuration
# ======================
class Config:
    EMBEDDING_MODEL = "text-embedding-3-large"
    EMBEDDING_DIM = 3072
    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 256
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    VALIDATION_THRESHOLD = 0.95
    ACCEPTED_MIME_TYPES = {
        'application/pdf', 'text/plain',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'image/png', 'image/jpeg', 'text/html'
    }
    MAX_INDEXES = 5


# ======================
# System Initialization
# ======================
def init_system() -> None:
    """Initialize API clients, Pinecone, and ML models."""
    # Ensure required environment variables are set
    if not os.getenv("PINECONE_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        sys.exit("Missing required API key(s): Check PINECONE_API_KEY and OPENAI_API_KEY")

    global oai_client, pc, reranker, device
    oai_client = OpenAI()
    pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        reranker = AutoModelForSequenceClassification.from_pretrained(Config.RERANKER_MODEL).to(device)
        reranker.tokenizer = AutoTokenizer.from_pretrained(Config.RERANKER_MODEL)
    except Exception as e:
        sys.exit(f"Model loading failed: {e}")


# ======================
# Helper Functions
# ======================
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


# ======================
# File Processor
# ======================
class FileProcessor:
    """Class to handle file processing based on MIME type."""

    def __init__(self) -> None:
        self.textract_config = {
            'pdftotext': {'layout': True},
            'ocr': {'language': 'eng'}
        }

    def process_file(self, path: Path) -> Optional[str]:
        """Process a file based on its MIME type."""
        try:
            mime = magic.from_file(str(path), mime=True)
            if mime not in Config.ACCEPTED_MIME_TYPES:
                logging.info(f"Skipping unsupported MIME type: {mime}")
                return None
            return self._dispatch_processing(path, mime)
        except Exception as e:
            logging.error(f"Error processing {path}: {e}")
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
                        pix = page.get_pixmap(dpi=300)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        text += "\n" + pytesseract.image_to_string(img)
            return text
        except Exception as e:
            logging.error(f"PDF processing error ({path}): {e}")
            return None

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


# ======================
# Text Processor
# ======================
class TextProcessor:
    """Class for text splitting and chunking using langchain's RecursiveCharacterTextSplitter."""

    def __init__(self) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len
        )

    def chunk_text(self, text: str, metadata: dict) -> List[Any]:
        """Split text into chunks and attach a content hash to each chunk."""
        if not text:
            return []
        chunks = self.splitter.create_documents([text], [metadata])
        for chunk in chunks:
            chunk.metadata['content_hash'] = hashlib.sha256(chunk.page_content.encode()).hexdigest()
        return chunks


# ======================
# Embeddings Generator
# ======================
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


# ======================
# Guided RAG System GUI
# ======================
class GuidedRAGInterface(tk.Tk):
    """Tkinter-based GUI for the Universal RAG System."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Universal RAG System")
        self.geometry("1200x800")
        self.queue = queue.Queue()

        # State variables
        self.selected_index = tk.StringVar()
        self.selected_namespace = tk.StringVar()
        self.setup_complete = False
        self.current_step = "welcome"

        # Initialize processors
        self.file_processor = FileProcessor()
        self.text_processor = TextProcessor()

        # Build UI
        self._create_widgets()
        self.after(100, self._process_queue)
        self.start_guided_setup()

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

    def _toggle_ui_state(self, enabled: bool) -> None:
        """Enable or disable UI elements."""
        state = "normal" if enabled else "disabled"
        widgets = [self.dir_btn, self.search_btn, self.query_entry,
                   self.index_combo, self.namespace_combo, self.refresh_indexes_btn,
                   self.refresh_namespaces_btn, self.new_index_btn, self.new_namespace_btn]
        for widget in widgets:
            widget.config(state=state)
        self.status.config(text="Ready" if enabled else "Processing...")

    def _process_queue(self) -> None:
        """Process queued UI updates from background threads."""
        while not self.queue.empty():
            try:
                task = self.queue.get()
                task()
            except Exception as e:
                self.status.config(text=f"Queue error: {e}")
                logging.error(f"Error processing queue: {traceback.format_exc()}")
        self.after(100, self._process_queue)

    def _queue_ui_task(self, task) -> None:
        """Helper to add a task to the UI update queue."""
        self.queue.put(task)

    # -------------------------
    # Index / Namespace Handling
    # -------------------------
    def create_new_index(self) -> None:
        """Prompt user to create a new index and create it if valid."""
        new_index = simpledialog.askstring("New Index", "Enter a name for the new index:")
        if not new_index:
            messagebox.showwarning("Input Required", "Please enter an index name.")
            return

        if new_index.isdigit():
            new_index = f"index-{new_index}"

        raw = pc.list_indexes()
        existing = extract_index_names(raw)
        if new_index in existing:
            messagebox.showinfo("Index Exists", f"Index '{new_index}' already exists. Selecting it.")
            self.selected_index.set(new_index)
            self.update_pc_index(new_index)
            return

        if len(existing) >= Config.MAX_INDEXES:
            messagebox.showerror("Index Creation Error", f"Cannot create new index. Max of {Config.MAX_INDEXES} reached.")
            return

        try:
            pc.create_index(
                name=new_index,
                dimension=Config.EMBEDDING_DIM,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
            )
            if wait_for_index_ready(new_index):
                messagebox.showinfo("Index Created", f"Index '{new_index}' is ready.")
                self.refresh_index_list()
                self.selected_index.set(new_index)
                self.update_pc_index(new_index)
            else:
                messagebox.showerror("Timeout", f"Index '{new_index}' was not ready in time.")
        except Exception as e:
            messagebox.showerror("Index Creation Error", str(e))

    def create_new_namespace(self) -> None:
        """Prompt user to create a new namespace."""
        new_namespace = simpledialog.askstring("New Namespace", "Enter a name for the new namespace:")
        if new_namespace:
            current = list(self.namespace_combo['values'])
            if new_namespace not in current:
                current.append(new_namespace)
                self.namespace_combo['values'] = current
            self.selected_namespace.set(new_namespace)
            messagebox.showinfo("Namespace Set", f"Namespace set to '{new_namespace}'.")

    def refresh_index_list(self) -> None:
        """Refresh the list of indexes from Pinecone."""
        try:
            raw = pc.list_indexes()
            indexes = extract_index_names(raw)
            if not indexes:
                messagebox.showinfo("No Indexes", "No indexes found. Please create one.")
                self.index_combo['values'] = []
                self.selected_index.set("")
                global pc_index
                pc_index = None
                return

            self.index_combo['values'] = indexes
            if self.selected_index.get() not in indexes and indexes:
                self.selected_index.set(indexes[0])
                self.update_pc_index(indexes[0])
            self.refresh_namespace_list()
        except Exception as e:
            messagebox.showerror("Index Refresh Error", f"Error refreshing indexes: {e}")

    def update_pc_index(self, index_name: str) -> None:
        """Update the global Pinecone index object."""
        global pc_index
        try:
            pc_index = pc.Index(index_name)
        except Exception as e:
            messagebox.showerror("Index Update Error", str(e))

    def refresh_namespace_list(self) -> None:
        """Refresh the namespace list from the current Pinecone index."""
        try:
            if not pc_index:
                self.namespace_combo['values'] = []
                self.selected_namespace.set("")
                return

            stats = pc_index.describe_index_stats()
            if isinstance(stats, dict) and "namespaces" in stats:
                ns = list(stats.get("namespaces", {}).keys())
            else:
                try:
                    ns = list(stats.namespaces.keys()) if hasattr(stats, 'namespaces') else []
                except (AttributeError, TypeError):
                    ns = []

            self.namespace_combo['values'] = ns
            if self.selected_namespace.get() not in ns and ns:
                self.selected_namespace.set(ns[0])
        except Exception as e:
            messagebox.showerror("Namespace Refresh Error", f"Error refreshing namespaces: {e}")

    def on_index_change(self, event: tk.Event) -> None:
        """Handle index selection change."""
        index_name = self.selected_index.get()
        self.update_pc_index(index_name)
        self.refresh_namespace_list()

    # -------------------------
    # Document Processing
    # -------------------------
    def _select_directory(self) -> None:
        """Open a dialog for selecting a directory and process its files."""
        path = filedialog.askdirectory()
        if path:
            self._toggle_ui_state(False)
            threading.Thread(target=self._process_files, args=(path,), daemon=True).start()

    def _process_files(self, path: str) -> None:
        """Process all files in the selected directory and upsert embeddings."""
        try:
            file_paths = []
            for root, _, files in os.walk(path):
                file_paths.extend([Path(root) / f for f in files])

            total = len(file_paths)
            processed = 0
            vectors = []
            namespace = self.selected_namespace.get() or None

            for idx, file_path in enumerate(file_paths):
                text = self.file_processor.process_file(file_path)
                if text:
                    chunks = self.text_processor.chunk_text(text, {"source": str(file_path)})
                    embeddings = generate_embeddings([c.page_content for c in chunks])
                    if embeddings:
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
                processed += 1
                prog_val = (processed / total) * 100
                self._queue_ui_task(lambda v=prog_val: self.progress.configure(value=v))
                self._queue_ui_task(lambda p=processed: self.status.configure(text=f"Processed {p}/{total} files"))

            # Upsert in batches of 100 vectors
            for i in range(0, len(vectors), 100):
                pc_index.upsert(vectors=vectors[i:i+100], namespace=namespace)

            self._queue_ui_task(lambda: messagebox.showinfo("Processing Complete", f"Indexed {len(vectors)} chunks from {processed} files"))
        except Exception as e:
            self._queue_ui_task(lambda: messagebox.showerror("Processing Error", str(e)))
        finally:
            self._queue_ui_task(lambda: self._toggle_ui_state(True))

    # -------------------------
    # Searching
    # -------------------------
    def _execute_search(self) -> None:
        """Initiate a search query from user input."""
        query = self.query_entry.get().strip()
        if not query:
            messagebox.showwarning("Empty Query", "Please enter a search term")
            return
        self._toggle_ui_state(False)
        threading.Thread(target=self._perform_search, args=(query,), daemon=True).start()

    def _perform_search(self, query: str) -> None:
        """Perform the search and rerank results using the reranker model."""
        try:
            namespace = self.selected_namespace.get() or None
            emb_list = generate_embeddings([query])
            if not emb_list:
                raise ValueError("Embedding generation returned None.")
            emb = emb_list[0]

            results = pc_index.query(
                vector=emb,
                top_k=50,
                include_metadata=True,
                namespace=namespace
            )

            # Reranking with the cross-encoder model
            pairs = [(query, hit.metadata.get('text', '')) for hit in results.matches]
            inputs = reranker.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to(device)
            logits = reranker(**inputs).logits

            if logits.shape[1] >= 2:
                scores = torch.softmax(logits, dim=1)[:, 1].cpu().detach().numpy()
            else:
                scores = torch.sigmoid(logits).squeeze(-1).cpu().detach().numpy()

            sorted_results = sorted(zip(results.matches, scores), key=lambda x: x[1], reverse=True)[:10]

            # Use dict.get() with default values to handle missing metadata keys
            context = "\n\n".join([
                f"Source: {res.metadata.get('source', 'Unknown')}\n{res.metadata.get('text', 'No content available')}"
                for res, _ in sorted_results
            ])

            answer = oai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"Answer using ONLY:\n{context}"},
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                max_tokens=1000
            ).choices[0].message.content

            result_text = f"Query: {query}\n\nAnswer: {answer}\n\nSources:\n"
            for idx, (match, score) in enumerate(sorted_results):
                result_text += f"{idx+1}: \"{match.metadata.get('source', 'Unknown')}\" (Score: {score:.3f})\n"

            self._queue_ui_task(lambda: self.results.delete(1.0, tk.END))
            self._queue_ui_task(lambda rt=result_text: self.results.insert(tk.END, rt))
        except Exception as e:
            error_msg = str(e)
            trace = traceback.format_exc()
            self._queue_ui_task(lambda em=error_msg, tb=trace: messagebox.showerror(
                "Search Error",
                f"Error during search:\n{em}\n\nTechnical details:\n{tb}"
            ))
        finally:
            self._queue_ui_task(lambda: self._toggle_ui_state(True))

    # -------------------------
    # Wizard Flow
    # -------------------------
    def initialize_system(self) -> None:
        """Initialize the system components."""
        try:
            init_system()
        except Exception as e:
            self._queue_ui_task(lambda: messagebox.showerror("Initialization Error", str(e)))

    def start_guided_setup(self) -> None:
        """Begin the setup wizard for the RAG system."""
        self.initialize_system()
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

    def handle_back(self) -> None:
        """Handle the Back button action based on current step."""
        if self.current_step in ["select_index", "create_index", "setup_complete"]:
            self.show_index_selection()

    def handle_next(self) -> None:
        """Handle the Next button action based on current step."""
        if self.current_step == "index_selection":
            choice = self.setup_choice.get()
            if choice == "create":
                self.show_create_index()
            elif choice == "select":
                self.show_select_index()
            else:
                messagebox.showwarning("Selection Required", "Please select an option to continue.")
        elif self.current_step == "select_index":
            if not self.setup_listbox.curselection():
                messagebox.showwarning("Selection Required", "Please select an index.")
                return
            index_name = self.setup_listbox.get(self.setup_listbox.curselection()[0])
            self.selected_index.set(index_name)
            self.update_pc_index(index_name)
            self.show_namespace_selection()
        elif self.current_step == "create_index":
            index_name = self.setup_entry.get().strip()
            if not index_name:
                messagebox.showwarning("Input Required", "Please enter an index name.")
                return
            self.create_index_and_continue(index_name)
        elif self.current_step == "select_namespace":
            if not self.setup_listbox.curselection():
                messagebox.showwarning("Selection Required", "Please select a namespace.")
                return
            ns_display = self.setup_listbox.get(self.setup_listbox.curselection()[0])
            namespace = ns_display if ns_display != "(default)" else ""
            self.selected_namespace.set(namespace)
            self.show_setup_complete()
        elif self.current_step == "create_namespace":
            namespace = self.setup_entry.get().strip()
            if not namespace:
                messagebox.showwarning("Input Required", "Please enter a namespace.")
                return
            self.selected_namespace.set(namespace)
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
            raw = pc.list_indexes()
            indexes = extract_index_names(raw)
            if not indexes:
                messagebox.showinfo("No Indexes", "No existing indexes found. Please create one.")
                self.show_create_index()
                return
            for idx_name in indexes:
                self.setup_listbox.insert(tk.END, idx_name)
            if self.setup_listbox.size() > 0:
                self.setup_listbox.selection_set(0)
        except Exception as e:
            messagebox.showerror("Index List Error", f"Error listing indexes: {e}")
            self.show_index_selection()

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

    def create_index_and_continue(self, index_name: str) -> None:
        """Create a new index and proceed to the namespace selection."""
        try:
            raw = pc.list_indexes()
            existing = extract_index_names(raw)
            if index_name in existing:
                messagebox.showinfo("Index Exists", f"Index '{index_name}' already exists. Selecting it.")
                self.selected_index.set(index_name)
                self.update_pc_index(index_name)
                self.show_namespace_selection()
                return

            if len(existing) >= Config.MAX_INDEXES:
                messagebox.showerror("Index Creation Error", f"Cannot create new index. Maximum of {Config.MAX_INDEXES} indexes reached.")
                return

            if index_name.isdigit():
                index_name = f"index-{index_name}"

            try:
                pc.create_index(
                    name=index_name,
                    dimension=Config.EMBEDDING_DIM,
                    metric="cosine",
                    spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
                )
                if wait_for_index_ready(index_name):
                    messagebox.showinfo("Index Created", f"Index '{index_name}' is ready for use.")
                    self.selected_index.set(index_name)
                    self.update_pc_index(index_name)
                    self.show_namespace_selection()
                else:
                    messagebox.showerror("Timeout Error", f"Index '{index_name}' did not become ready in time.")
            except Exception as e:
                if "409" in str(e) or "ALREADY_EXISTS" in str(e):
                    messagebox.showinfo("Index Exists", f"Index '{index_name}' already exists. Selecting it.")
                    self.selected_index.set(index_name)
                    self.update_pc_index(index_name)
                    self.show_namespace_selection()
                else:
                    messagebox.showerror("Index Creation Error", str(e))
        except Exception as e:
            messagebox.showerror("Index Creation Error", str(e))

    def show_namespace_selection(self) -> None:
        """Proceed directly to setup completion (namespace selection skipped for simplicity)."""
        self.show_setup_complete()

    def show_setup_complete(self) -> None:
        """Complete the setup and switch to the main application UI."""
        self.current_step = "setup_complete"
        self.setup_frame.grid_remove()
        self.app_frame.grid()
        self.setup_complete = True
        self.refresh_index_list()
        self.refresh_namespace_list()


# ======================
# Main Execution
# ======================
def main() -> None:
    """Main entry point for the application."""
    # Check for required packages
    try:
        import magic
        import textract
    except ImportError:
        print("Missing required packages. Install with:")
        print("  brew install libmagic tesseract poppler")
        print("  pip install python-magic textract")
        sys.exit(1)

    try:
        init_system()
    except Exception as e:
        sys.exit(f"Initialization failed: {e}")

    global pc_index
    pc_index = None

    app = GuidedRAGInterface()
    app.mainloop()


if __name__ == "__main__":
    main()
    
