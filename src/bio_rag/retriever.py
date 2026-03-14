from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS


@dataclass
class RetrievedPassage:
    rank: int
    score: float
    qid: str
    text: str
    source_question: str
    source_answer: str
    authors: str = ""
    year: str = ""
    journal: str = ""
    title: str = ""


class BioRetriever:
    """Hybrid retriever: FAISS vector search + keyword re-ranking."""

    def __init__(self, vectorstore: FAISS, top_k: int = 5) -> None:
        self.vectorstore = vectorstore
        self.top_k = top_k

    def retrieve(self, question: str) -> list[RetrievedPassage]:
        # Fetch more candidates then re-rank
        fetch_k = self.top_k * 4
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            question, k=fetch_k
        )

        candidates: list[RetrievedPassage] = []
        for i, (doc, score) in enumerate(docs_and_scores, start=1):
            candidates.append(
                RetrievedPassage(
                    rank=i,
                    score=float(score),
                    qid=str(doc.metadata.get("qid", "")),
                    text=doc.page_content,
                    source_question=str(doc.metadata.get("question", "")),
                    source_answer=str(doc.metadata.get("answer", "")),
                    authors=str(doc.metadata.get("authors", "")),
                    year=str(doc.metadata.get("year", "")),
                    journal=str(doc.metadata.get("journal", "")),
                    title=str(doc.metadata.get("title", "")),
                )
            )

        # Re-rank with keyword overlap boost
        reranked = _rerank(question, candidates)
        # Take top_k after re-ranking
        final = reranked[: self.top_k]
        # Reassign ranks
        for i, p in enumerate(final, start=1):
            p.rank = i
        return final


def _rerank(
    question: str, passages: list[RetrievedPassage]
) -> list[RetrievedPassage]:
    """Re-rank passages using combined vector score + keyword overlap."""
    q_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower()))
    stopwords = {
        "does", "with", "that", "this", "from", "have", "been", "were",
        "what", "which", "their", "they", "than", "more", "about", "the",
        "and", "for", "are", "not", "but", "can", "all", "has",
    }
    q_keywords = q_words - stopwords
    if not q_keywords:
        q_keywords = q_words

    scored: list[tuple[float, RetrievedPassage]] = []
    # Normalise L2 scores: lower is better, convert to 0-1 similarity
    if not passages:
        return []
    max_score = max(p.score for p in passages)
    min_score = min(p.score for p in passages)
    score_range = max_score - min_score if max_score > min_score else 1.0

    for p in passages:
        # Vector similarity (0-1, higher=better)
        vec_sim = 1.0 - (p.score - min_score) / score_range

        # Keyword overlap with title + source_question (most important)
        title_q = (p.title + " " + p.source_question).lower()
        title_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', title_q))
        title_overlap = len(q_keywords & title_words) / max(len(q_keywords), 1)

        # Keyword overlap with full text
        text_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', p.text.lower()))
        text_overlap = len(q_keywords & text_words) / max(len(q_keywords), 1)

        # Combined score: vector_sim * 0.3 + title_match * 0.5 + text_match * 0.2
        combined = vec_sim * 0.3 + title_overlap * 0.5 + text_overlap * 0.2
        scored.append((combined, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored]
