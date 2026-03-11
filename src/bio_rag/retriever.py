from __future__ import annotations

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
    def __init__(self, vectorstore: FAISS, top_k: int = 5) -> None:
        self.vectorstore = vectorstore
        self.top_k = top_k

    def retrieve(self, question: str) -> list[RetrievedPassage]:
        docs_and_scores = self.vectorstore.similarity_search_with_score(question, k=self.top_k)
        passages: list[RetrievedPassage] = []

        for i, (doc, score) in enumerate(docs_and_scores, start=1):
            passages.append(
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

        return passages
