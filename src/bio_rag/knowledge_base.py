from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .config import BioRAGConfig
from .data_loader import PubMedQASample


class KnowledgeBaseBuilder:
    def __init__(self, config: BioRAGConfig) -> None:
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)

    def build(self, samples: list[PubMedQASample]) -> FAISS:
        documents = [
            Document(
                page_content=f"{sample.title}\n{sample.question}\n{sample.context}",
                metadata={
                    "qid": sample.qid,
                    "question": sample.question,
                    "answer": sample.answer,
                    "authors": sample.authors,
                    "year": sample.year,
                    "journal": sample.journal,
                    "title": sample.title,
                },
            )
            for sample in samples
        ]
        return FAISS.from_documents(documents, self.embeddings)

    def save(self, vectorstore: FAISS) -> None:
        self.config.index_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(self.config.index_path))

    def load_or_build(self, samples: list[PubMedQASample]) -> FAISS:
        path = self.config.index_path
        if _looks_like_faiss_index(path):
            return FAISS.load_local(
                str(path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

        vectorstore = self.build(samples)
        self.save(vectorstore)
        return vectorstore


def _looks_like_faiss_index(path: Path) -> bool:
    return path.exists() and (path / "index.faiss").exists() and (path / "index.pkl").exists()
