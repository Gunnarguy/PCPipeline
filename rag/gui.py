import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import threading
import queue
import traceback
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import font as tkfont
import darkdetect

# Import from processing module
from rag.processing import (
    init_system, wait_for_index_ready, extract_index_names,
    FileProcessor, TextProcessor, generate_embeddings,
    device
)
from rag.config import Config
import torch
import numpy as np
# Import Pinecone types if needed for type hinting
# Using 'Any' as fallback if pinecone types aren't directly available/resolvable
from typing import Any  # Keep Any for fallback
try:
    # Attempt to import specific types for better hinting
    from pinecone import Index as PineconeIndexType
    # Use Any as robust fallback for response/vector types
    PineconeQueryResponseType = Any
    PineconeScoredVectorType = Any
except ImportError:
    PineconeIndexType = Any
    PineconeQueryResponseType = Any
    PineconeScoredVectorType = Any


# --- Helper Functions ---

def get_pinecone_client():
    """Get or initialize the Pinecone client."""
    from rag.processing import pc as current_pc
    if current_pc is None:
        # Initialize system if not already done
        init_system()
        from rag.processing import pc as current_pc
        if current_pc is None:
            raise ValueError("Failed to initialize Pinecone client")
    return current_pc


def get_openai_client():
    """Get the current OpenAI client from the processing module."""
    # Import oai_client locally to avoid top-level unused/redefined issues
    from rag.processing import oai_client
    if oai_client is None:
        # Try to initialize the system if not done already
        init_system()
        # Re-import after potential initialization
        from rag.processing import oai_client
        if oai_client is None:
            raise ValueError("Failed to initialize OpenAI client")
    return oai_client


class GuidedRAGInterface(tk.Tk):
    """Tkinter-based GUI for the Universal RAG System."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Universal RAG System")
        self.geometry("1400x800")
        self.queue: queue.Queue = queue.Queue()

        # State variables
        self.selected_index = tk.StringVar()
        self.selected_namespace = tk.StringVar()
        self.setup_complete = False
        self.current_step = "welcome"

        # Initialize processors
        self.file_processor = FileProcessor()
        self.text_processor = TextProcessor()
        self.pc_index: Optional[PineconeIndexType] = None  # Manage index instance

        # Add placeholders for attributes causing Pylance errors / used in UI
        self.sample_text = tk.StringVar()
        self.viz_mime_type = tk.StringVar()
        self.viz_frame: Optional[ttk.Frame] = None  # Will be created later

        # UI Elements (declare upfront for clarity)
        self.main_frame: Optional[ttk.Frame] = None
        self.setup_frame: Optional[ttk.Frame] = None
        self.app_frame: Optional[ttk.Frame] = None
        self.setup_title: Optional[ttk.Label] = None
        self.setup_instructions: Optional[ttk.Label] = None
        self.options_frame: Optional[ttk.Frame] = None
        self.button_frame: Optional[ttk.Frame] = None
        self.back_btn: Optional[ttk.Button] = None
        self.next_btn: Optional[ttk.Button] = None
        self.dir_btn: Optional[ttk.Button] = None
        self.index_combo: Optional[ttk.Combobox] = None
        self.new_index_btn: Optional[ttk.Button] = None
        self.refresh_indexes_btn: Optional[ttk.Button] = None
        self.namespace_combo: Optional[ttk.Combobox] = None
        self.new_namespace_btn: Optional[ttk.Button] = None
        self.refresh_namespaces_btn: Optional[ttk.Button] = None
        self.progress: Optional[ttk.Progressbar] = None
        self.status: Optional[ttk.Label] = None
        self.query_entry: Optional[ttk.Entry] = None
        self.search_btn: Optional[ttk.Button] = None
        self.results_notebook: Optional[ttk.Notebook] = None
        self.results: Optional[tk.Text] = None
        self.log_text: Optional[tk.Text] = None
        self.log_font: Optional[tkfont.Font] = None
        self.autoscroll_var = tk.BooleanVar(value=True)
        self.autoscroll_check: Optional[ttk.Checkbutton] = None
        self.clear_log_btn: Optional[ttk.Button] = None
        self.log_menu: Optional[tk.Menu] = None
        self.setup_choice = tk.StringVar()
        self.radio_frame: Optional[ttk.Frame] = None
        self.setup_listbox: Optional[tk.Listbox] = None
        self.setup_scrollbar: Optional[ttk.Scrollbar] = None
        self.setup_entry: Optional[ttk.Entry] = None

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
        self.main_frame.grid(row=0, column=0, rowspan=3,
                             sticky="nsew", padx=10, pady=10)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(2, weight=1)

        # Setup (wizard) frame
        self.setup_frame = ttk.Frame(self.main_frame)
        self.setup_frame.grid(row=0, column=0, sticky="nsew")
        self.setup_frame.columnconfigure(0, weight=1)

        self.setup_title = ttk.Label(
            self.setup_frame, text="Welcome to Universal RAG System",
            font=("Arial", 14, "bold")
        )
        self.setup_title.grid(row=0, column=0, pady=20)

        self.setup_instructions = ttk.Label(
            self.setup_frame, text="", wraplength=800, justify="center"
        )
        self.setup_instructions.grid(row=1, column=0, pady=10)

        self.options_frame = ttk.Frame(self.setup_frame)
        self.options_frame.grid(row=2, column=0, pady=20)

        self.button_frame = ttk.Frame(self.setup_frame)
        self.button_frame.grid(row=3, column=0, pady=20)

        self.back_btn = ttk.Button(self.button_frame, text="Back",
                                   command=self.handle_back)
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
        if not self.app_frame:
            logging.error("App frame not initialized in _create_app_controls")
            return

        # Control Frame
        control_frame = ttk.Frame(self.app_frame)
        control_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        control_frame.columnconfigure(tuple(range(9)), weight=1)

        self.dir_btn = ttk.Button(
            control_frame, text="Select Documents",
            command=self._select_directory
        )
        self.dir_btn.grid(row=0, column=0, padx=5, pady=2)

        ttk.Label(control_frame, text="Index:").grid(
            row=0, column=1, padx=5, pady=2
        )
        self.index_combo = ttk.Combobox(
            control_frame, textvariable=self.selected_index,
            state="readonly", width=20
        )
        self.index_combo.grid(row=0, column=2, padx=5, pady=2)
        self.index_combo.bind("<<ComboboxSelected>>", self.on_index_change)

        self.new_index_btn = ttk.Button(
            control_frame, text="New Index", command=self.create_new_index
        )
        self.new_index_btn.grid(row=0, column=3, padx=5, pady=2)

        self.refresh_indexes_btn = ttk.Button(
            control_frame, text="Refresh Indexes",
            command=self.refresh_index_list
        )
        self.refresh_indexes_btn.grid(row=0, column=4, padx=5, pady=2)

        ttk.Label(control_frame, text="Namespace:").grid(
            row=0, column=5, padx=5, pady=2
        )
        self.namespace_combo = ttk.Combobox(
            control_frame, textvariable=self.selected_namespace,
            state="readonly", width=20
        )
        self.namespace_combo.grid(row=0, column=6, padx=5, pady=2)

        self.new_namespace_btn = ttk.Button(
            control_frame, text="New Namespace",
            command=self.create_new_namespace
        )
        self.new_namespace_btn.grid(row=0, column=7, padx=5, pady=2)

        self.refresh_namespaces_btn = ttk.Button(
            control_frame, text="Refresh Namespaces",
            command=self.refresh_namespace_list
        )
        self.refresh_namespaces_btn.grid(row=0, column=8, padx=5, pady=2)

        self.progress = ttk.Progressbar(
            control_frame, orient="horizontal", mode="determinate"
        )
        self.progress.grid(row=1, column=0, columnspan=4,
                           sticky="ew", padx=5, pady=2)

        self.status = ttk.Label(control_frame, text="Ready")
        self.status.grid(row=1, column=4, columnspan=5, padx=5, pady=2)

        # Query Frame
        query_frame = ttk.Frame(self.app_frame)
        query_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        query_frame.columnconfigure(0, weight=1)

        self.query_entry = ttk.Entry(query_frame)
        self.query_entry.grid(row=0, column=0, sticky="ew", padx=5)
        self.search_btn = ttk.Button(
            query_frame, text="Search", command=self._execute_search
        )
        self.search_btn.grid(row=0, column=1, padx=5)

        # Create a notebook with tabs for results and processing log only
        self.results_notebook = ttk.Notebook(self.app_frame)
        self.results_notebook.grid(row=2, column=0, sticky="nsew",
                                   padx=10, pady=5)

        # Set dark mode detection
        self.is_dark_mode = darkdetect.isDark() if hasattr(darkdetect, 'isDark') else False

        # Set appropriate colors based on system theme
        self.bg_color = "#1e1e1e" if self.is_dark_mode else "#f8f8f8"
        self.text_color = "#ffffff" if self.is_dark_mode else "#000000"
        self.highlight_bg = "#2d2d30" if self.is_dark_mode else "#ffffd0"

        # Results Frame (first tab)
        results_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(results_frame, text="Results")
        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)

        self.results = tk.Text(
            results_frame,
            wrap=tk.WORD,
            bg=self.bg_color,
            fg=self.text_color,
            insertbackground=self.text_color,  # Cursor color
            selectbackground="#264f78",       # Selection background
            selectforeground=self.text_color  # Selection text color
        )
        self.results.grid(row=0, column=0, sticky="nsew")
        results_scrollbar = ttk.Scrollbar(
            results_frame, orient="vertical", command=self.results.yview
        )
        results_scrollbar.grid(row=0, column=1, sticky="ns")
        self.results.config(yscrollcommand=results_scrollbar.set)

        # Enhanced Processing Log Frame (second tab)
        log_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(log_frame, text="Processing Log")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        # Use a consistent monospace font
        self.log_font = tkfont.Font(family="Courier", size=10)

        # Create Text widget with fixed width font and padding
        self.log_text = tk.Text(
            log_frame,
            wrap=tk.WORD,
            background=self.bg_color,
            foreground=self.text_color,
            insertbackground=self.text_color,
            selectbackground="#264f78",
            selectforeground=self.text_color,
            font=self.log_font,
            padx=10,
            pady=10,
            spacing1=1,  # Space before paragraph
            spacing2=0,  # Space between wrapped lines
            spacing3=1   # Space after paragraph
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scrollbar = ttk.Scrollbar(
            log_frame, orient="vertical", command=self.log_text.yview
        )
        log_scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.config(yscrollcommand=log_scrollbar.set)

        # Define tags with colors
        self._update_theme_tags() # Apply initial theme tags

        # Add auto-scroll checkbox
        self.autoscroll_check = ttk.Checkbutton(
            log_frame, text="Auto-scroll", variable=self.autoscroll_var
        )
        self.autoscroll_check.grid(row=1, column=0, sticky="w", padx=5, pady=2)

        # Add a clear log button
        self.clear_log_btn = ttk.Button(
            log_frame, text="Clear Log",
            command=lambda: self.log_text.delete("1.0", tk.END) if self.log_text else None
        )
        self.clear_log_btn.grid(row=1, column=0, sticky="e", padx=5, pady=2)

        # Add right-click menu for the log
        self.log_menu = tk.Menu(self.log_text, tearoff=0)
        self.log_menu.add_command(label="Copy", command=self._copy_log)
        self.log_menu.add_command(
            label="Clear",
            command=lambda: self.log_text.delete("1.0", tk.END) if self.log_text else None
        )
        self.log_text.bind("<Button-3>", self._show_log_menu)

    def _visualize_chunking(self):
        """Visualize how text would be chunked and tokenized."""
        text = self.sample_text.get()
        if not text:
            text = ("This is a sample text to visualize how chunking and "
                    "tokenization work.")
            self.sample_text.set(text) # Use .set() for StringVar

        mime_type = self.viz_mime_type.get() or "text/plain"

        # Ensure viz_frame exists and is a widget before clearing
        if isinstance(self.viz_frame, tk.Widget):
            for widget in self.viz_frame.winfo_children():
                widget.destroy()
        else:
            # Create viz_frame if it doesn't exist or isn't a widget
            if not self.app_frame: return # Should not happen if UI is built
            self.viz_frame = ttk.Frame(self.app_frame)
            # Example placement - adjust row/column as needed for layout
            self.viz_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)

        # Frame for tokenization
        token_frame = ttk.LabelFrame(
            self.viz_frame, text="Tokenization Visualization"
        )
        token_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Get token visualization
        token_viz = self.text_processor.visualize_chunk(text)

        # Display token details
        token_details = ttk.Frame(token_frame)
        token_details.pack(fill="x", expand=False, padx=5, pady=5)

        ttk.Label(token_details, text=f"Character count: {token_viz['character_count']}").pack(anchor="w")
        ttk.Label(token_details, text=f"Token count: {token_viz['token_count']}").pack(anchor="w")
        ttk.Label(token_details, text=f"Tokens per character: {token_viz['tokens_per_char']:.2f}").pack(anchor="w")

        # Show tokens in a scrollable text widget
        ttk.Label(token_frame, text="Individual Tokens:").pack(anchor="w", padx=5)
        token_text = tk.Text(token_frame, height=5, wrap=tk.WORD)
        token_text.pack(fill="both", expand=True, padx=5, pady=5)
        token_scroll = ttk.Scrollbar(
            token_text, orient="vertical", command=token_text.yview
        )
        token_scroll.pack(side="right", fill="y")
        token_text.config(yscrollcommand=token_scroll.set)

        # Insert tokens with alternating colors
        token_text.tag_configure("odd", background="#f5f5f5")
        token_text.tag_configure("even", background="#e0e0e0")

        for i, token in enumerate(token_viz["tokens"]):
            tag = "odd" if i % 2 else "even"
            token_text.insert(tk.END, f"[{token}]", tag)

        # Generate chunks - Ensure mime_type is not None and is a string
        if not isinstance(mime_type, str):
             mime_type = "text/plain" # Default if None
        # Ensure metadata is passed correctly if needed by chunk_text
        chunks, analytics = self.text_processor.chunk_text(
            text, {"source": "visualization"}, mime_type
        )

        # Create chunking info frame
        chunk_frame = ttk.LabelFrame(self.viz_frame, text="Chunking Analysis")
        chunk_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Show chunking strategy and stats
        ttk.Label(chunk_frame, text=f"Chunking Strategy: {analytics['chunk_strategy']}").pack(anchor="w", padx=5)
        ttk.Label(chunk_frame, text=f"Total Chunks: {analytics['total_chunks']}").pack(anchor="w", padx=5)
        ttk.Label(chunk_frame, text=f"Average Tokens per Chunk: {analytics['avg_tokens_per_chunk']:.1f}").pack(anchor="w", padx=5)

        if analytics["total_chunks"] > 0 and analytics.get("token_distribution"): # Check key exists
            # Create figure for visualizing chunk token distribution
            fig = Figure(figsize=(8, 3), dpi=100)
            ax = fig.add_subplot(111)
            token_distribution = analytics.get("token_distribution", [])
            if token_distribution: # Check if list is not empty
                ax.hist(token_distribution,
                        bins=min(10, len(token_distribution)),
                        alpha=0.7)
            ax.set_title("Token Count Distribution")
            ax.set_xlabel("Tokens per Chunk")
            ax.set_ylabel("Frequency")

            # Add the plot to the frame
            canvas = FigureCanvasTkAgg(fig, master=chunk_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        # Display all chunks
        if chunks:
            chunks_text_frame = ttk.LabelFrame(self.viz_frame, text="Resulting Chunks")
            chunks_text_frame.pack(fill="both", expand=True, padx=10, pady=10)

            chunks_display = tk.Text(chunks_text_frame, wrap=tk.WORD, height=10)
            chunks_display.pack(fill="both", expand=True, padx=5, pady=5)
            chunks_scroll = ttk.Scrollbar(
                chunks_display, orient="vertical", command=chunks_display.yview
            )
            chunks_scroll.pack(side="right", fill="y")
            chunks_display.config(yscrollcommand=chunks_scroll.set)

            for i, chunk in enumerate(chunks):
                chunk_header = (f"--- Chunk {i+1} ({len(chunk.page_content)} chars, "
                                f"~{self.text_processor.count_tokens(chunk.page_content)} tokens) ---\n")
                chunks_display.insert(tk.END, chunk_header, "chunk_header")
                chunks_display.insert(tk.END, f"{chunk.page_content}\n\n")

            chunks_display.tag_configure("chunk_header", foreground="blue")

    def _show_log_menu(self, event):
        """Show the context menu for the log text area."""
        if self.log_menu:
            self.log_menu.post(event.x_root, event.y_root)

    def _copy_log(self):
        """Copy selected text or all text from log to clipboard."""
        if not self.log_text: return
        try:
            selected_text = self.log_text.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            selected_text = self.log_text.get(1.0, tk.END)

        if selected_text:
            self.clipboard_clear()
            self.clipboard_append(selected_text)

    def log_message(self, message: str, level: str = "INFO") -> None:
        """Add a message to the log with timestamp and proper tag."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        # Determine tag based on level
        tag_map = {
            "ERROR": "error", "WARNING": "warning", "INFO": "info",
            "SUCCESS": "success"
        }
        tag = tag_map.get(level.upper())

        # Format the message with timestamp
        formatted_msg = f"[{timestamp}] [{level}]: {message}\n"

        # Queue the UI update to add the message with the appropriate tag
        self._queue_ui_task(lambda msg=formatted_msg, t=tag: self._safe_insert_text(msg, t))

    def _safe_insert_text(self, text: str, tag: Optional[str] = None) -> None:
        """A safer method to insert text with tags that avoids rendering issues."""
        if not self.log_text: return

        # Get current position
        start_index = self.log_text.index(tk.END + "-1c") # Before the last newline

        # Insert the text
        self.log_text.insert(tk.END, text)

        # Apply the tag if provided
        if tag:
            end_index = self.log_text.index(tk.END + "-1c") # Exclude the last newline
            try:
                # Ensure start index is before end index
                if self.log_text.compare(start_index, "<", end_index):
                    self.log_text.tag_add(tag, start_index, end_index)
            except tk.TclError as e:
                # If we get a TCL error, don't crash - just log it
                logging.error(f"Error applying tag '{tag}' from {start_index} to {end_index}: {e}")

        # Auto-scroll after insertion is complete
        if self.autoscroll_var.get():
            self.log_text.see(tk.END)
            self.log_text.update_idletasks()

    def _toggle_ui_state(self, enabled: bool) -> None:
        """Enable or disable UI elements."""
        state = "normal" if enabled else "disabled"
        widgets = [self.dir_btn, self.search_btn, self.query_entry,
                   self.index_combo, self.namespace_combo,
                   self.refresh_indexes_btn, self.refresh_namespaces_btn,
                   self.new_index_btn, self.new_namespace_btn]
        for widget in widgets:
            if widget: # Check if widget exists
                widget.config(state=state)
        if self.status:
            self.status.config(text="Ready" if enabled else "Processing...")

    def _process_queue(self) -> None:
        """Process queued UI updates from background threads."""
        while not self.queue.empty():
            try:
                task = self.queue.get_nowait()
                task()
            except queue.Empty:
                pass
            except Exception as e:
                if self.status:
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
        new_index = simpledialog.askstring("New Index",
                                           "Enter a name for the new index:")
        if not new_index:
            messagebox.showwarning("Input Required", "Please enter an index name.")
            return

        if new_index.isdigit():
            new_index = f"index-{new_index}"

        try:
            pc = get_pinecone_client()
            raw = pc.list_indexes()
            existing = extract_index_names(raw)
            if new_index in existing:
                messagebox.showinfo("Index Exists",
                                    f"Index '{new_index}' already exists. Selecting it.")
                self.selected_index.set(new_index)
                self.update_pc_index(new_index)
                return

            if len(existing) >= Config.MAX_INDEXES:
                messagebox.showerror("Index Creation Error",
                                     f"Cannot create new index. Max of {Config.MAX_INDEXES} reached.")
                return

            # Create an index using the newer API style
            pc.create_index(
                name=new_index,
                dimension=Config.EMBEDDING_DIM,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
            )

            if wait_for_index_ready(new_index):
                messagebox.showinfo("Index Created",
                                    f"Index '{new_index}' is ready.")
                self.refresh_index_list()
                self.selected_index.set(new_index)
                self.update_pc_index(new_index)
            else:
                messagebox.showerror("Timeout",
                                     f"Index '{new_index}' was not ready in time.")
        except Exception as e:
            messagebox.showerror("Index Creation Error", str(e))

    def create_new_namespace(self) -> None:
        """Prompt user to create a new namespace."""
        new_namespace = simpledialog.askstring("New Namespace",
                                               "Enter a name for the new namespace:")
        if new_namespace and self.namespace_combo:
            current = list(self.namespace_combo['values'])
            if new_namespace not in current:
                current.append(new_namespace)
                self.namespace_combo['values'] = current
            self.selected_namespace.set(new_namespace)
            messagebox.showinfo("Namespace Set",
                                f"Namespace set to '{new_namespace}'.")

    def refresh_index_list(self) -> None:
        """Refresh the list of indexes from Pinecone."""
        if not self.index_combo: return
        try:
            pc = get_pinecone_client()
            raw = pc.list_indexes()
            indexes = extract_index_names(raw)
            if not indexes:
                messagebox.showinfo("No Indexes",
                                    "No indexes found. Please create one.")
                self.index_combo['values'] = []
                self.selected_index.set("")
                self.pc_index = None # Clear local index reference
                return

            self.index_combo['values'] = indexes
            if self.selected_index.get() not in indexes and indexes:
                self.selected_index.set(indexes[0])
                self.update_pc_index(indexes[0])
            self.refresh_namespace_list()
        except Exception as e:
            messagebox.showerror("Index Refresh Error",
                                 f"Error refreshing indexes: {e}")

    def update_pc_index(self, index_name: str) -> None:
        """Update the Pinecone index instance for this GUI object."""
        try:
            pc = get_pinecone_client()
            # Use the recommended method based on pinecone-client version
            # Newer versions use pc.Index(...)
            if hasattr(pc, 'Index'):
                 self.pc_index = pc.Index(index_name)
            # Older versions might use pc.index(...) - keep for compatibility
            elif hasattr(pc, 'index') and callable(pc.index): # Check if callable
                 logging.warning("Using potentially deprecated pc.index(). Consider updating pinecone-client.")
                 self.pc_index = pc.index(index_name)
            else:
                 raise AttributeError("Pinecone client object has no callable 'Index' or 'index' attribute.")
            logging.info(f"Pinecone index set to: {index_name}")
        except Exception as e:
            self.pc_index = None # Ensure index is None on error
            messagebox.showerror("Index Update Error",
                                 f"Failed to connect to index '{index_name}': {e}")

    def refresh_namespace_list(self) -> None:
        """Refresh the namespace list from the current Pinecone index."""
        if not self.namespace_combo: return
        try:
            # Use the class instance's index
            if not self.pc_index:
                self.namespace_combo['values'] = []
                self.selected_namespace.set("")
                return

            stats = self.pc_index.describe_index_stats()
            # Handle potential variations in stats object structure
            ns: List[str] = []
            if hasattr(stats, 'namespaces') and stats.namespaces:
                 ns = list(stats.namespaces.keys())
            elif isinstance(stats, dict) and "namespaces" in stats and stats["namespaces"]:
                 ns = list(stats["namespaces"].keys())

            self.namespace_combo['values'] = ns
            if self.selected_namespace.get() not in ns and ns:
                self.selected_namespace.set(ns[0])
        except Exception as e:
            messagebox.showerror("Namespace Refresh Error",
                                 f"Error refreshing namespaces: {e}")

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
            threading.Thread(target=self._process_files, args=(path,),
                             daemon=True).start()

    def _process_files(self, path: str) -> None:
        """Process files and upsert embeddings with detailed visualization."""
        try:
            # Switch to log tab when processing starts
            if self.results_notebook:
                self._queue_ui_task(lambda: self.results_notebook.select(1))

            self.log_message(f"Starting document processing from path: {path}")

            # Use the class instance's index
            if not self.pc_index:
                self._queue_ui_task(lambda: messagebox.showerror(
                    "Processing Error",
                    "No Pinecone index selected or connected."
                ))
                self.log_message("Error: No Pinecone index selected/connected",
                                 "ERROR")
                return

            # List all files
            self.log_message("Scanning directory for files...")
            file_paths = []
            for root, _, files in os.walk(path):
                file_paths.extend([Path(root) / f for f in files])

            total = len(file_paths)
            self.log_message(f"Found {total} files to process")

            processed = 0
            successful = 0
            skipped = 0
            failed = 0
            vectors: List[Dict[str, Any]] = []
            namespace = self.selected_namespace.get() or None

            self.log_message(f"Using namespace: {namespace or 'default'}")

            # Process each file
            for idx, file_path in enumerate(file_paths):
                self._log_file_start(
                    f"Processing file {processed+1}/{total}: {file_path.name}"
                )

                # Process file
                text, mime_type = self.file_processor.process_file(file_path)
                if text:
                    char_count = len(text)
                    line_count = text.count('\n') + 1
                    estimated_tokens = char_count // 4
                    self.log_message(
                        f"✓ Extracted text: {char_count} chars, "
                        f"~{estimated_tokens} tokens, {line_count} lines "
                        f"(MIME: {mime_type})", "SUCCESS"
                    )

                    sample = "\n".join(text.split("\n")[:3])
                    if len(sample) > 100:
                        sample = sample[:100] + "..."
                    self._log_sample(f"Text sample: {sample}")

                    # Chunking
                    if not isinstance(mime_type, str):
                        mime_type = "text/plain"
                    self.log_message(
                        f"⊕ Chunking text using {mime_type} strategy...", "INFO"
                    )
                    chunks, analytics = self.text_processor.chunk_text(
                        text, {"source": str(file_path)}, mime_type
                    )
                    chunk_count = len(chunks)
                    self._log_chunking_stats(chunks, analytics)

                    # Generate embeddings
                    self.log_message(
                        f"⊗ Generating embeddings for {chunk_count} chunks...",
                        "INFO"
                    )
                    embeddings = generate_embeddings(
                        [c.page_content for c in chunks]
                    )

                    if embeddings:
                        successful += 1
                        dim = len(embeddings[0]) if embeddings else 0
                        self.log_message(
                            f"✓ Generated {len(embeddings)} embeddings "
                            f"({dim} dimensions each)", "SUCCESS"
                        )
                        self._log_vectors_creation(chunks, str(file_path), idx)
                        vectors.extend(
                            self._create_vectors(chunks, embeddings,
                                                 str(file_path), idx)
                        )
                    else:
                        failed += 1
                        self.log_message("✗ Failed to generate embeddings",
                                         "ERROR")
                else:
                    skipped += 1
                    if mime_type:
                        self.log_message(
                            f"⚠ Skipped: Could not extract text (MIME: {mime_type})",
                            "WARNING"
                        )
                    else:
                        self.log_message("⚠ Skipped: Unknown file type",
                                         "WARNING")

                self._log_divider()
                processed += 1
                prog_val = (processed / total) * 100
                if self.progress:
                    self._queue_ui_task(lambda v=prog_val: self.progress.config(value=v))
                # Correctly capture loop variables for the lambda
                current_processed = processed
                current_successful = successful
                current_skipped = skipped
                current_failed = failed
                status_text = (f"Processed: {current_processed}/{total} (Success: {current_successful}, "
                               f"Skipped: {current_skipped}, Failed: {current_failed})")
                if self.status:
                    # Pass captured variables to the lambda
                    self._queue_ui_task(lambda st=status_text: self.status.config(text=st))


            # Batch uploads
            if vectors:
                self._log_upsert_start(len(vectors))
                batch_size = 100
                batch_count = (len(vectors) + batch_size - 1) // batch_size

                for i in range(0, len(vectors), batch_size):
                    batch_num = (i // batch_size) + 1
                    batch = vectors[i:i+batch_size]
                    self._log_batch_upsert(batch_num, batch_count, len(batch))
                    if self.pc_index: # Check index exists before upsert
                        self.pc_index.upsert(vectors=batch, namespace=namespace)

                self.log_message("✓ All vectors successfully upserted to Pinecone",
                                 "SUCCESS")
                self._log_final_summary(processed, successful, skipped, failed,
                                        len(vectors))
            else:
                self.log_message("⚠ No vectors were created. Nothing to upsert.",
                                 "WARNING")
                self._queue_ui_task(lambda: messagebox.showwarning(
                    "Processing Complete",
                    "No content was indexed. Please check the log for details."
                ))

        except Exception as e:
            error_msg = str(e)
            trace = traceback.format_exc()
            self.log_message(f"✗ Error during processing: {error_msg}", "ERROR")
            self.log_message(f"Stack trace: {trace}", "ERROR")
            self._queue_ui_task(lambda err=error_msg: messagebox.showerror(
                "Processing Error", err
            ))
        finally:
            self.log_message("Processing task finished")
            self._queue_ui_task(lambda: self._toggle_ui_state(True))

    def _create_vectors(self, chunks: List[Any], embeddings: List[List[float]],
                        file_path: str, idx: int) -> List[Dict[str, Any]]:
        """Create vector objects from chunks and embeddings."""
        vectors = []
        for chunk, emb in zip(chunks, embeddings):
            # Ensure metadata exists and has expected keys
            metadata = getattr(chunk, 'metadata', {})
            content_hash = metadata.get('content_hash', f'chunk_{idx}_{len(vectors)}') # Fallback ID
            vector_id = f"{content_hash}_{idx}"
            vectors.append({
                "id": vector_id,
                "values": emb,
                "metadata": {
                    "text": getattr(chunk, 'page_content', ''),
                    "source": file_path,
                    "hash": content_hash
                }
            })
        return vectors

    # Helper methods for enhanced logging with visualization
    def _log_file_start(self, message: str) -> None:
        """Log the start of processing a file with highlighting."""
        header = f"\n{message}\n" + "=" * len(message) + "\n"
        self._queue_ui_task(lambda msg=header: self._safe_insert_text(msg, "file_header"))

    def _log_sample(self, text: str) -> None:
        """Log a sample of text with styling."""
        self._queue_ui_task(lambda t=f"{text}\n": self._safe_insert_text(t, "highlight"))

    def _log_chunking_stats(self, chunks: List[Any], analytics: Dict[str, Any]) -> None:
        """Log chunking statistics with a visual representation."""
        chunk_count = len(chunks)
        avg_tokens = analytics.get("avg_tokens_per_chunk", 0)

        stats_text = f"✓ Created {chunk_count} chunks (avg. {avg_tokens:.1f} tokens per chunk)\n"
        self._queue_ui_task(lambda t=stats_text: self._safe_insert_text(t, "success"))

        # Create a simple visualization
        if chunk_count > 0 and "token_distribution" in analytics:
            dist = analytics.get("token_distribution", [])
            if dist: # Check if distribution is not empty
                min_val = min(dist)
                max_val = max(dist)
                dist_text = f"Token distribution: {min_val} - {max_val} tokens\n"
                for i, count in enumerate(sorted(dist)):
                    if i > 5:
                        dist_text += "... and more\n"
                        break
                    dist_text += f"  - {count} tokens\n"
                self._queue_ui_task(lambda t=dist_text: self._safe_insert_text(t, "chunk_info"))

            # Show a simple summary of chunks
            if chunks:
                example_text = "\nExample chunks:\n"
                for i, chunk in enumerate(chunks[:2]):
                    text_content = getattr(chunk, 'page_content', '')
                    if len(text_content) > 40:
                        text_content = text_content[:37] + "..."
                    example_text += f"  Chunk {i+1}: {text_content}\n"

                if len(chunks) > 2:
                    example_text += f"  ... and {len(chunks) - 2} more chunks\n"

                self._queue_ui_task(lambda t=example_text: self._safe_insert_text(t, "chunk_info"))

    def _log_vectors_creation(self, chunks: List[Any], file_path: str, idx: int) -> None:
        """Visualize the vector creation process."""
        text = (f"Creating vectors for {len(chunks)} chunks from "
                f"{os.path.basename(file_path)}...\n")
        self._queue_ui_task(lambda t=text: self._safe_insert_text(t, "info"))

    def _log_upsert_start(self, vector_count: int) -> None:
        """Log the start of the upsert process."""
        divider = "=" * 50
        text = f"\n{divider}\nUPSERTING {vector_count} VECTORS TO PINECONE\n{divider}\n"
        self._queue_ui_task(lambda t=text: self._safe_insert_text(t, "highlight"))

    def _log_batch_upsert(self, batch_num: int, total_batches: int, batch_size: int) -> None:
        """Log batch upsert with progress visualization."""
        percent = int((batch_num / total_batches) * 100) if total_batches > 0 else 0
        text = (f"Batch {batch_num}/{total_batches} ({batch_size} vectors) - "
                f"{percent}% complete\n")
        self._queue_ui_task(lambda t=text: self._safe_insert_text(t, "info"))

    def _log_final_summary(self, processed: int, successful: int, skipped: int,
                           failed: int, vector_count: int) -> None:
        """Log the final summary with visual emphasis."""
        divider = "=" * 30
        summary = (
            f"\n{divider}\n"
            f"PROCESSING COMPLETE\n"
            f"Files processed: {processed}\n"
            f"Successfully processed: {successful}\n"
            f"Skipped: {skipped}\n"
            f"Failed: {failed}\n"
            f"Total vectors indexed: {vector_count}\n"
            f"{divider}\n"
        )
        self._queue_ui_task(lambda t=summary: self._safe_insert_text(t, "highlight"))

        # Show dialog after updating the log
        success_msg = f"Processing complete: {vector_count} vectors indexed"
        self._queue_ui_task(lambda msg=success_msg: messagebox.showinfo(
            "Processing Complete", msg
        ))

    def _log_divider(self) -> None:
        """Add a visual divider in the log."""
        divider = "\n" + "-" * 40 + "\n\n"
        self._queue_ui_task(lambda t=divider: self._safe_insert_text(t))

    # -------------------------
    # Searching
    # -------------------------
    def _execute_search(self) -> None:
        """Initiate a search query from user input."""
        if not self.query_entry: return
        query = self.query_entry.get().strip()
        if not query:
            messagebox.showwarning("Empty Query", "Please enter a search term")
            return
        self._toggle_ui_state(False)
        threading.Thread(target=self._perform_search, args=(query,),
                         daemon=True).start()

    def _perform_search(self, query: str) -> None:
        """Perform the search and rerank results using the reranker model."""
        try:
            # Use the class instance's index
            if not self.pc_index:
                self._queue_ui_task(lambda: messagebox.showerror(
                    "Search Error", "No Pinecone index selected or connected."
                ))
                self._queue_ui_task(lambda: self._toggle_ui_state(True))
                return

            # Get the namespace from UI
            namespace = self.selected_namespace.get() or None

            # Get OpenAI client
            try:
                oai_client = get_openai_client()
            except ValueError as client_error: # Catch specific error
                self._queue_ui_task(lambda err=str(client_error): messagebox.showerror(
                    "Search Error", f"OpenAI client not available: {err}"
                ))
                self._queue_ui_task(lambda: self._toggle_ui_state(True))
                return

            emb_list = generate_embeddings([query])
            if not emb_list:
                raise ValueError("Embedding generation returned None.")
            emb = emb_list[0]

            # Use the class instance's index
            results: Optional[PineconeQueryResponseType] = self.pc_index.query(
                vector=emb,
                top_k=50, # Retrieve more for reranking
                include_metadata=True,
                namespace=namespace
            )

            # Initialize scores list
            scores: List[float] = []
            matches_list: List[PineconeScoredVectorType] = []
            # Import reranker locally
            from rag.processing import reranker, reranker_tokenizer

            # Check if reranker and tokenizer are both available
            if reranker is not None and reranker_tokenizer is not None:
                # Reranking with the cross-encoder model
                try:
                    # Ensure results and results.matches exist and are iterable
                    matches_to_rerank = []
                    if results and hasattr(results, 'matches') and results.matches:
                        matches_to_rerank = results.matches
                    else:
                        logging.warning("No matches found in Pinecone results.")

                    if not matches_to_rerank:
                         scores = []
                    else:
                        # Ensure metadata text is treated as string
                        pairs = [(query, str(hit.metadata.get('text', '')))
                                 for hit in matches_to_rerank if hit.metadata]
                        if pairs:
                             try:
                                 inputs = reranker_tokenizer(
                                     pairs, padding=True, truncation=True,
                                     return_tensors="pt"
                                 ).to(device)
                                 with torch.no_grad(): # Inference mode
                                     logits = reranker(**inputs).logits

                                 # Check logits shape before processing
                                 if logits.ndim > 1 and logits.shape[1] >= 2:
                                     scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
                                 else:
                                     scores = torch.sigmoid(logits).squeeze(-1).cpu().numpy().tolist()
                             except Exception as tokenizer_error:
                                 logging.error(f"Tokenizer/Reranker error: {tokenizer_error}. Falling back.")
                                 scores = np.linspace(0.99, 0.80, len(matches_to_rerank)).tolist()
                        else:
                             scores = []

                    if self.status:
                        self._queue_ui_task(lambda: self.status.config(
                            text="Reranked results using cross-encoder model"
                        ))
                except Exception as rerank_error:
                    logging.error(f"Reranking process failed: {rerank_error}. Using default scores.")
                    # Ensure matches_list is defined in this scope
                    matches_list = results.matches if results and hasattr(results, 'matches') and results.matches else []
                    scores = np.linspace(0.99, 0.80, len(matches_list)).tolist()
            else:
                 # Fallback if reranker isn't available
                 matches_list = results.matches if results and hasattr(results, 'matches') and results.matches else []
                 if self.status:
                     self._queue_ui_task(lambda: self.status.config(
                         text="Reranker not available. Using default ranking."
                     ))
                 scores = np.linspace(0.99, 0.80, len(matches_list)).tolist()

            # Ensure matches_list is defined before sorting
            matches_list = results.matches if results and hasattr(results, 'matches') and results.matches else []

            # Sort results by scores, ensuring lists have same length
            sorted_results: List[Tuple[PineconeScoredVectorType, float]] = []
            if len(matches_list) == len(scores):
                 combined = list(zip(matches_list, scores))
                 sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
                 sorted_results = sorted_combined[:10]
            elif matches_list:
                 logging.warning(f"Score length mismatch ({len(scores)}) vs matches ({len(matches_list)}). Using default scores.")
                 sorted_results = [(match, 0.0) for match in matches_list[:10]]

            # Generate context string safely
            context_parts = []
            for res, _ in sorted_results:
                 source = res.metadata.get('source', 'Unknown') if res.metadata else 'Unknown'
                 text_content = res.metadata.get('text', 'No content available') if res.metadata else 'No content available'
                 context_parts.append(f"Source: {source}\n{text_content}")
            context = "\n\n".join(context_parts)

            # Generate answer
            answer = "Error: Could not generate answer."
            if context:
                try:
                    completion = oai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": f"Answer using ONLY the following context:\n{context}"},
                            {"role": "user", "content": query}
                        ],
                        temperature=0.3,
                        max_tokens=1000
                    )
                    if completion.choices:
                        answer = completion.choices[0].message.content or answer
                except Exception as llm_error:
                    error_msg = str(llm_error)
                    self._queue_ui_task(lambda err=error_msg: messagebox.showerror(
                        "LLM Error", f"Failed to generate answer: {err}"
                    ))
            else:
                 answer = "No relevant context found to generate an answer."

            result_text = f"Query: {query}\n\nAnswer: {answer}\n\nSources:\n"
            for idx, (match, score) in enumerate(sorted_results):
                source = match.metadata.get('source', 'Unknown') if match and match.metadata else 'Unknown'
                result_text += f"{idx+1}: \"{source}\" (Score: {score:.3f})\n"

            if self.results:
                self._queue_ui_task(lambda: self.results.delete(1.0, tk.END))
                self._queue_ui_task(lambda rt=result_text: self.results.insert(tk.END, rt))
        except Exception as search_error: # Catch broader search errors
            error_msg = str(search_error)
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
            error_msg = str(e)
            self._queue_ui_task(lambda err=error_msg: messagebox.showerror(
                "Initialization Error", err
            ))

    def start_guided_setup(self) -> None:
        """Begin the setup wizard for the RAG system."""
        self.initialize_system()
        self.current_step = "welcome"
        self.setup_complete = False

        if self.app_frame: self.app_frame.grid_remove()
        if self.setup_frame: self.setup_frame.grid()

        if self.setup_title:
            self.setup_title.config(text="Welcome to Universal RAG System")
        if self.setup_instructions:
            self.setup_instructions.config(
                text="This wizard will guide you through setting up your RAG system.\n"
                     "First, you'll need to select or create an index."
            )

        if self.options_frame:
            for widget in self.options_frame.winfo_children():
                widget.grid_remove()

        if self.back_btn: self.back_btn.config(state="disabled")
        if self.next_btn:
            self.next_btn.config(text="Start Setup",
                                 command=self.show_index_selection)

    def show_index_selection(self) -> None:
        """Display options for index selection."""
        self.current_step = "index_selection"
        if self.setup_title:
            self.setup_title.config(text="Index Selection")
        if self.setup_instructions:
            self.setup_instructions.config(
                text="Do you want to create a new index or select an existing one?"
            )

        if not self.options_frame: return
        for widget in self.options_frame.winfo_children():
            widget.grid_remove()

        self.setup_choice = tk.StringVar()
        self.radio_frame = ttk.Frame(self.options_frame)
        self.radio_frame.grid(row=0, column=0)

        create_radio = ttk.Radiobutton(
            self.radio_frame, text="Create new index", value="create",
            variable=self.setup_choice
        )
        create_radio.grid(row=0, column=0, pady=5, sticky="w")

        select_radio = ttk.Radiobutton(
            self.radio_frame, text="Select existing index", value="select",
            variable=self.setup_choice
        )
        select_radio.grid(row=1, column=0, pady=5, sticky="w")

        if self.back_btn: self.back_btn.config(state="normal")
        if self.next_btn:
            self.next_btn.config(text="Next", command=self.handle_next)

    def handle_back(self) -> None:
        """Handle the Back button action based on current step."""
        if self.current_step in ["select_index", "create_index",
                                 "select_namespace", "create_namespace",
                                 "setup_complete"]:
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
                messagebox.showwarning("Selection Required",
                                       "Please select an option to continue.")
        elif self.current_step == "select_index":
            if not hasattr(self, 'setup_listbox') or not self.setup_listbox or not self.setup_listbox.curselection():
                messagebox.showwarning("Selection Required",
                                       "Please select an index.")
                return
            index_name = self.setup_listbox.get(self.setup_listbox.curselection()[0])
            self.selected_index.set(index_name)
            self.update_pc_index(index_name)
            self.show_namespace_selection()
        elif self.current_step == "create_index":
            if not hasattr(self, 'setup_entry') or not self.setup_entry: return
            index_name = self.setup_entry.get().strip()
            if not index_name:
                messagebox.showwarning("Input Required",
                                       "Please enter an index name.")
                return
            self.create_index_and_continue(index_name)
        elif self.current_step == "select_namespace":
            if not hasattr(self, 'setup_listbox') or not self.setup_listbox or not self.setup_listbox.curselection():
                messagebox.showwarning("Selection Required",
                                       "Please select a namespace.")
                return
            ns_display = self.setup_listbox.get(self.setup_listbox.curselection()[0])
            namespace = ns_display if ns_display != "(default)" else ""
            self.selected_namespace.set(namespace)
            self.show_setup_complete()
        elif self.current_step == "create_namespace":
            if not hasattr(self, 'setup_entry') or not self.setup_entry: return
            namespace = self.setup_entry.get().strip()
            if not namespace:
                messagebox.showwarning("Input Required",
                                       "Please enter a namespace.")
                return
            self.selected_namespace.set(namespace)
            self.show_setup_complete()

    def show_select_index(self) -> None:
        """Display a list for selecting an existing index."""
        self.current_step = "select_index"
        if self.setup_title:
            self.setup_title.config(text="Select Existing Index")
        if self.setup_instructions:
            self.setup_instructions.config(
                text="Select an index from the list below."
            )

        if not self.options_frame: return
        for widget in self.options_frame.winfo_children():
            widget.grid_remove()

        self.setup_listbox = tk.Listbox(self.options_frame, width=40, height=10)
        self.setup_listbox.grid(row=0, column=0, pady=10)
        self.setup_scrollbar = ttk.Scrollbar(
            self.options_frame, orient="vertical",
            command=self.setup_listbox.yview
        )
        self.setup_listbox.config(yscrollcommand=self.setup_scrollbar.set)
        self.setup_scrollbar.grid(row=0, column=1, sticky="ns")

        try:
            pc = get_pinecone_client()
            raw = pc.list_indexes()
            indexes = extract_index_names(raw)
            if not indexes:
                messagebox.showinfo("No Indexes",
                                    "No existing indexes found. Please create one.")
                self.show_create_index()
                return
            for idx_name in indexes:
                self.setup_listbox.insert(tk.END, idx_name)
            if self.setup_listbox.size() > 0:
                self.setup_listbox.selection_set(0)
        except Exception as e:
            messagebox.showerror("Index List Error",
                                 f"Error listing indexes: {e}")
            self.show_index_selection()

    def show_create_index(self) -> None:
        """Display UI for creating a new index."""
        self.current_step = "create_index"
        if self.setup_title:
            self.setup_title.config(text="Create New Index")
        if self.setup_instructions:
            self.setup_instructions.config(
                text="Enter a name for your new index."
            )

        if not self.options_frame: return
        for widget in self.options_frame.winfo_children():
            widget.grid_remove()

        self.setup_entry = ttk.Entry(self.options_frame, width=40)
        self.setup_entry.grid(row=0, column=0, pady=10)
        if self.next_btn:
            self.next_btn.config(
                text="Create",
                command=lambda: self.create_index_and_continue(self.setup_entry.get() if self.setup_entry else "")
            )

    def create_index_and_continue(self, index_name: str) -> None:
        """Create a new index and proceed to the namespace selection."""
        try:
            pc = get_pinecone_client()
            raw = pc.list_indexes()
            existing = extract_index_names(raw)
            if index_name in existing:
                messagebox.showinfo("Index Exists",
                                    f"Index '{index_name}' already exists. Selecting it.")
                self.selected_index.set(index_name)
                self.update_pc_index(index_name)
                self.show_namespace_selection()
                return

            if len(existing) >= Config.MAX_INDEXES:
                messagebox.showerror("Index Creation Error",
                                     f"Cannot create new index. Maximum of {Config.MAX_INDEXES} indexes reached.")
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
                    messagebox.showinfo("Index Created",
                                        f"Index '{index_name}' is ready for use.")
                    self.selected_index.set(index_name)
                    self.update_pc_index(index_name)
                    self.show_namespace_selection()
                else:
                    messagebox.showerror("Timeout Error",
                                         f"Index '{index_name}' did not become ready in time.")
            except Exception as e:
                if "409" in str(e) or "ALREADY_EXISTS" in str(e):
                    messagebox.showinfo("Index Exists",
                                        f"Index '{index_name}' already exists. Selecting it.")
                    self.selected_index.set(index_name)
                    self.update_pc_index(index_name)
                    self.show_namespace_selection()
                else:
                    messagebox.showerror("Index Creation Error", str(e))
        except Exception as e:
            messagebox.showerror("Index Creation Error", str(e))

    def show_namespace_selection(self) -> None:
        """Display UI for selecting or creating a namespace."""
        self.current_step = "select_namespace"
        if self.setup_title:
            self.setup_title.config(text="Select Namespace")
        if self.setup_instructions:
            self.setup_instructions.config(
                text="Select a namespace or create a new one."
            )

        if not self.options_frame: return
        for widget in self.options_frame.winfo_children():
            widget.grid_remove()

        # Frame for namespace options
        ns_options_frame = ttk.Frame(self.options_frame)
        ns_options_frame.grid(row=0, column=0, pady=10)

        # Listbox for existing namespaces
        ttk.Label(ns_options_frame, text="Existing Namespaces:").grid(
            row=0, column=0, sticky="w"
        )
        self.setup_listbox = tk.Listbox(ns_options_frame, width=40, height=5)
        self.setup_listbox.grid(row=1, column=0, pady=5)
        self.setup_scrollbar = ttk.Scrollbar(
            ns_options_frame, orient="vertical",
            command=self.setup_listbox.yview
        )
        self.setup_listbox.config(yscrollcommand=self.setup_scrollbar.set)
        self.setup_scrollbar.grid(row=1, column=1, sticky="ns")

        # Populate listbox
        self.setup_listbox.insert(tk.END, "(default)") # Add default option
        try:
            if self.pc_index:
                stats = self.pc_index.describe_index_stats()
                ns: List[str] = []
                if hasattr(stats, 'namespaces') and stats.namespaces:
                     ns = list(stats.namespaces.keys())
                elif isinstance(stats, dict) and "namespaces" in stats and stats["namespaces"]:
                     ns = list(stats["namespaces"].keys())

                for namespace in ns:
                    self.setup_listbox.insert(tk.END, namespace)
            if self.setup_listbox.size() > 0:
                 self.setup_listbox.selection_set(0) # Select default initially
        except Exception as e:
            messagebox.showerror("Namespace List Error",
                                 f"Error listing namespaces: {e}")

        # Option to create new namespace
        ttk.Button(ns_options_frame, text="Create New Namespace",
                   command=self.show_create_namespace).grid(row=2, column=0, pady=10)

        if self.next_btn:
            self.next_btn.config(text="Use Selected Namespace",
                                 command=self.handle_next)

    def show_create_namespace(self) -> None:
        """Display UI for creating a new namespace."""
        self.current_step = "create_namespace"
        if self.setup_title:
            self.setup_title.config(text="Create New Namespace")
        if self.setup_instructions:
            self.setup_instructions.config(
                text="Enter a name for the new namespace."
            )

        if not self.options_frame: return
        for widget in self.options_frame.winfo_children():
            widget.grid_remove()

        self.setup_entry = ttk.Entry(self.options_frame, width=40)
        self.setup_entry.grid(row=0, column=0, pady=10)
        if self.next_btn:
            self.next_btn.config(text="Create and Use", command=self.handle_next)

    def show_setup_complete(self) -> None:
        """Complete the setup and switch to the main application UI."""
        self.current_step = "setup_complete"
        if self.setup_frame: self.setup_frame.grid_remove()
        if self.app_frame: self.app_frame.grid()
        self.setup_complete = True
        self.refresh_index_list()
        self.refresh_namespace_list()

    # --- Theme Update Methods ---
    def _check_theme_change(self):
        """Check if the system theme has changed and update colors."""
        current_dark_mode = darkdetect.isDark() if hasattr(darkdetect, 'isDark') else False
        if current_dark_mode != self.is_dark_mode:
            self.is_dark_mode = current_dark_mode
            self._update_theme()

    def _update_theme(self):
        """Update the UI theme based on the current system theme."""
        self.bg_color = "#1e1e1e" if self.is_dark_mode else "#f8f8f8"
        self.text_color = "#ffffff" if self.is_dark_mode else "#000000"

        # Update text widgets if they exist
        if self.results:
            self.results.config(
                bg=self.bg_color, fg=self.text_color,
                insertbackground=self.text_color
            )
        if self.log_text:
            self.log_text.config(
                background=self.bg_color, foreground=self.text_color,
                insertbackground=self.text_color
            )
            self._update_theme_tags() # Re-apply tags

    def _update_theme_tags(self):
        """Update log text tag colors based on theme."""
        if not self.log_text: return
        if self.is_dark_mode:
            # Dark mode colors
            self.log_text.tag_config("error", foreground="#ff6b6b")
            self.log_text.tag_config("warning", foreground="#ffa94d")
            self.log_text.tag_config("info", foreground="#63b3ed")
            self.log_text.tag_config("success", foreground="#68d391")
            self.log_text.tag_config("highlight", background="#3b3b3b")
            self.log_text.tag_config("file_header", foreground="#4dabf7")
            self.log_text.tag_config("chunk_info", foreground="#9ae6b4")
        else:
            # Light mode colors
            self.log_text.tag_config("error", foreground="#e53e3e")
            self.log_text.tag_config("warning", foreground="#dd6b20")
            self.log_text.tag_config("info", foreground="#3182ce")
            self.log_text.tag_config("success", foreground="#38a169")
            self.log_text.tag_config("highlight", background="#fefcbf")
            self.log_text.tag_config("file_header", foreground="#0066cc")
            self.log_text.tag_config("chunk_info", foreground="#2f855a")

# --- Main Execution --- (Example, adjust as needed)
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO) # Add basic logging
#     app = GuidedRAGInterface()
#     # Optional: Start theme checking loop
#     # def theme_checker_loop():
#     #     app._check_theme_change() # Call instance method
#     #     app.after(1000, theme_checker_loop) # Check every second
#     # app.after(1000, theme_checker_loop)
#     app.mainloop()
