class Config:
    EMBEDDING_MODEL = "text-embedding-3-large"
    EMBEDDING_DIM = 3072
    DEFAULT_CHUNK_SIZE = 1024
    DEFAULT_CHUNK_OVERLAP = 256
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    VALIDATION_THRESHOLD = 0.95
    ACCEPTED_MIME_TYPES = {
        # Document formats
        'application/pdf', 'text/plain',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/msword',
        
        # Image formats
        'image/png', 'image/jpeg', 'image/gif', 'image/tiff', 'image/bmp',
        
        # Web formats
        'text/html', 'text/css',
        
        # Data formats
        'text/markdown', 'application/json', 'application/xml',
        'text/csv', 'text/tsv', 'text/rtf', 'application/rtf',
        
        # Code formats
        'application/x-python', 'text/x-python',
        'application/javascript', 'text/javascript', 
        
        # Generic binary - attempt to process if possible
        'application/octet-stream',
        
        # Office formats
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-powerpoint',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    }
    MAX_INDEXES = 5