from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DIABETES_KEYWORDS = [
    "diabetes",
    "blood glucose",
    "insulin",
    "type 1 diabetes",
    "type 2 diabetes",
    "diabetic complications",
]


@dataclass(frozen=True)
class BioRAGConfig:
    embedding_model: str = os.getenv(
        "BIO_RAG_EMBEDDING_MODEL", "dmis-lab/biobert-v1.1"
    )
    generator_model: str = os.getenv("BIO_RAG_GENERATOR_MODEL", "BioMistral/BioMistral-7B")
    index_path: Path = Path(os.getenv("BIO_RAG_INDEX_PATH", ".cache/bio_rag_faiss"))
    max_samples: int = int(os.getenv("BIO_RAG_MAX_SAMPLES", "2000"))
    top_k: int = int(os.getenv("BIO_RAG_TOP_K", "5"))
    claim_similarity_threshold: float = float(
        os.getenv("BIO_RAG_CLAIM_SIM_THRESHOLD", "0.62")
    )
    dataset_name: str = "qiaojin/PubMedQA"
