#!/usr/bin/env python3
# File: ultimate_rag_final.py
# Version: 5.0 (Production Ready)
import os
import sys
import hashlib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import magic
import textract
import fitz  # PyMuPDF
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'  # Explicit path
from PIL import Image
import pinecone
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import threading
import queue
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ======================
# Configuration
# ======================
class Config:
    EMBEDDING_MODEL = "text-embedding-3-large"
    EMBEDDING_DIM = 3072
    PINECONE_INDEX = "ultimate-rag-v6"
    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 256
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    VALIDATION_THRESHOLD = 0.95
    ACCEPTED_MIME_TYPES = {
        'application/pdf', 'text/plain', 
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'image/png', 'image/jpeg', 'text/html'
    }

# ======================
# System Initialization
# ======================
def init_system():
    # Validate environment
    assert os.getenv("PINECONE_API_KEY"), "Missing Pinecone API key"
    assert os.getenv("OPENAI_API_KEY"), "Missing OpenAI API key"
    
    global oai_client, pc_index, reranker, device
    oai_client = OpenAI()
    pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Pinecone index setup
    try:
        if Config.PINECONE_INDEX not in pc.list_indexes().names():
            pc.create_index(
                name=Config.PINECONE_INDEX,
                dimension=Config.EMBEDDING_DIM,
                metric="cosine",
                spec=pinecone.ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        pc_index = pc.Index(Config.PINECONE_INDEX)
    except Exception as e:
        sys.exit(f"Pinecone error: {str(e)}")
    
    # Model initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        reranker = AutoModelForSequenceClassification.from_pretrained(
            Config.RERANKER_MODEL
        ).to(device)
    except Exception as e:
        sys.exit(f"Model loading failed: {str(e)}")

# ======================
# File Processing
# ======================
class FileProcessor:
    def __init__(self):
        self.textract_config = {
            'pdftotext': {'layout': True},
            'ocr': {'language': 'eng'}
        }
    
    def process_file(self, path):
        try:
            mime = magic.from_file(path, mime=True)
            if mime not in Config.ACCEPTED_MIME_TYPES:
                return None
            return self._dispatch_processing(path, mime)
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            return None
    
    def _dispatch_processing(self, path, mime):
        if mime == 'application/pdf':
            return self._process_pdf(path)
        elif mime.startswith('image/'):
            return self._process_image(path)
        else:
            return self._process_generic(path)
    
    def _process_pdf(self, path):
        text = ""
        try:
            with fitz.open(path) as doc:
                for page in doc:
                    page_text = page.get_text("text")
                    text += page_text
                    if len(page_text.strip()) < 100:  # OCR fallback
                        pix = page.get_pixmap(dpi=300)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        text += "\n" + pytesseract.image_to_string(img)
            return text
        except Exception as e:
            print(f"PDF processing error: {str(e)}")
            return None
    
    def _process_image(self, path):
        try:
            return pytesseract.image_to_string(
                Image.open(path), 
                config='--psm 6 -c preserve_interword_spaces=1'
            )
        except Exception as e:
            print(f"Image processing error: {str(e)}")
            return None
    
    def _process_generic(self, path):
        try:
            return textract.process(path, **self.textract_config).decode('utf-8')
        except Exception as e:
            print(f"Text extraction error: {str(e)}")
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except:
                return None

# ======================
# Text Processing
# ======================
class TextProcessor:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len
        )
    
    def chunk_text(self, text, metadata):
        if not text:
            return []
        chunks = self.splitter.create_documents([text], [metadata])
        for chunk in chunks:
            chunk.metadata['content_hash'] = hashlib.sha256(
                chunk.page_content.encode()
            ).hexdigest()
        return chunks

# ======================
# Embedding Generation
# ======================
def generate_embeddings(texts):
    try:
        response = oai_client.embeddings.create(
            input=texts,
            model=Config.EMBEDDING_MODEL,
            dimensions=Config.EMBEDDING_DIM
        )
        return [e.embedding for e in response.data]
    except Exception as e:
        print(f"Embedding error: {str(e)}")
        return None

# ======================
# GUI Implementation
# ======================
class RAGInterface(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Universal RAG System")
        self.geometry("1200x800")
        self.queue = queue.Queue()
        self._create_widgets()
        self.after(100, self._process_queue)
        self.file_processor = FileProcessor()
        self.text_processor = TextProcessor()
    
    def _create_widgets(self):
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Control Panel
        control_frame = ttk.Frame(self)
        control_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        
        # Directory Selection
        self.dir_btn = ttk.Button(
            control_frame, 
            text="Select Documents", 
            command=self._select_directory
        )
        self.dir_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress Bar
        self.progress = ttk.Progressbar(
            control_frame, 
            orient="horizontal", 
            mode="determinate"
        )
        self.progress.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        # Status Label
        self.status = ttk.Label(control_frame, text="Ready")
        self.status.pack(side=tk.LEFT, padx=5)
        
        # Query Interface
        query_frame = ttk.Frame(self)
        query_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        
        self.query_entry = ttk.Entry(query_frame, width=100)
        self.query_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        self.search_btn = ttk.Button(
            query_frame, 
            text="Search", 
            command=self._execute_search
        )
        self.search_btn.pack(side=tk.LEFT, padx=5)
        
        # Results Display
        results_frame = ttk.Frame(self)
        results_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)
        
        self.results = tk.Text(results_frame, wrap=tk.WORD)
        self.results.grid(row=0, column=0, sticky="nsew")
        
        scrollbar = ttk.Scrollbar(
            results_frame, 
            orient="vertical", 
            command=self.results.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.results.config(yscrollcommand=scrollbar.set)
        
        # Configure grid weights
        self.rowconfigure(2, weight=1)
        self.columnconfigure(0, weight=1)
    
    def _select_directory(self):
        path = filedialog.askdirectory()
        if path:
            self._toggle_ui_state(False)
            threading.Thread(
                target=self._process_files, 
                args=(path,), 
                daemon=True
            ).start()
    
    def _process_files(self, path):
        try:
            file_paths = []
            for root, _, files in os.walk(path):
                file_paths.extend([Path(root) / f for f in files])
            
            total = len(file_paths)
            processed = 0
            vectors = []
            
            for idx, file_path in enumerate(file_paths):
                if text := self.file_processor.process_file(file_path):
                    chunks = self.text_processor.chunk_text(
                        text, 
                        {"source": str(file_path)}
                    )
                    if embeddings := generate_embeddings(
                        [c.page_content for c in chunks]
                    ):
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
                progress = (processed / total) * 100
                self.queue.put(lambda: self.progress.configure(value=progress))
                self.queue.put(lambda: self.status.configure(
                    text=f"Processed {processed}/{total} files"
                ))
            
            # Upsert in batches
            for i in range(0, len(vectors), 100):
                pc_index.upsert(vectors=vectors[i:i+100])
            
            self.queue.put(lambda: messagebox.showinfo(
                "Processing Complete",
                f"Indexed {len(vectors)} chunks from {processed} files"
            ))
            
        except Exception as e:
            self.queue.put(lambda: messagebox.showerror(
                "Processing Error",
                str(e)
            ))
        finally:
            self.queue.put(lambda: self._toggle_ui_state(True))
    
    def _execute_search(self):
        query = self.query_entry.get().strip()
        if not query:
            messagebox.showwarning("Empty Query", "Please enter a search term")
            return
        
        self._toggle_ui_state(False)
        threading.Thread(
            target=self._perform_search, 
            args=(query,), 
            daemon=True
        ).start()
    
    def _perform_search(self, query):
        try:
            # Generate query embedding
            emb = generate_embeddings([query])[0]
            
            # Pinecone search
            results = pc_index.query(
                vector=emb,
                top_k=50,
                include_metadata=True
            )
            
            # Rerank results
            pairs = [(query, hit.metadata['text']) for hit in results.matches]
            scores = torch.softmax(
                reranker(**reranker.tokenizer(
                    pairs, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt"
                ).to(device)).logits, 
                dim=1
            )[:, 1].cpu().detach().numpy()
            
            # Combine and sort
            sorted_results = sorted(
                zip(results.matches, scores),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # Generate answer
            context = "\n\n".join(
                [f"Source: {res.metadata['source']}\n{res.metadata['text']}" 
                for res, _ in sorted_results]
            )
            
            answer = oai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"Answer using ONLY:\n{context}"},
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                max_tokens=1000
            ).choices[0].message.content
            
            # Format results
            result_text = f"Query: {query}\n\nAnswer: {answer}\n\nSources:\n"
            for idx, (match, score) in enumerate(sorted_results):
                result_text += f"{idx+1}. {match.metadata['source']} (Score: {score:.3f})\n"
            
            self.queue.put(lambda: self.results.delete(1.0, tk.END))
            self.queue.put(lambda: self.results.insert(tk.END, result_text))
            
        except Exception as e:
            self.queue.put(lambda: messagebox.showerror(
                "Search Error",
                str(e)
            ))
        finally:
            self.queue.put(lambda: self._toggle_ui_state(True))
    
    def _toggle_ui_state(self, enabled):
        state = "normal" if enabled else "disabled"
        self.dir_btn.config(state=state)
        self.search_btn.config(state=state)
        self.query_entry.config(state=state)
        self.status.config(text="Ready" if enabled else "Processing...")
    
    def _process_queue(self):
        while not self.queue.empty():
            task = self.queue.get()
            task()
        self.after(100, self._process_queue)

# ======================
# Main Execution
# ======================
if __name__ == "__main__":
    # Verify system dependencies
    try:
        import magic
        import textract
    except ImportError:
        print("Missing required system packages. Install with:")
        print("  brew install libmagic tesseract poppler")
        print("  pip install python-magic textract")
        sys.exit(1)
    
    # Initialize system components
    try:
        init_system()
    except Exception as e:
        sys.exit(f"Initialization failed: {str(e)}")
    
    # Launch application
    app = RAGInterface()
    app.mainloop()
