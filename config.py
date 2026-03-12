import os

# Models (all local - no API key needed)
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"           # 33MB - text to vectors
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"       # 184MB - hallucination verification
GENERATOR_MODEL_NAME = "google/flan-t5-base"               # 990MB - answer generation

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_db")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "vector_db", "chroma_store")
DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_DIR = os.path.join(DATA_DIR, "raw_pdfs")

# Processing
TOP_K_RETRIEVE = 5
TOP_K_CANDIDATES = 15             # Broad retrieval before reranking
MIN_RELEVANCE_THRESHOLD = 0.50    # Minimum reranking similarity to accept results
FAITHFULNESS_THRESHOLD = 0.7
SOURCE_REJECTION_THRESHOLD = 0.15 # Below this faithfulness, hide sources entirely

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)