from __future__ import annotations

import logging
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer, util

from .retriever import RetrievedPassage

logger = logging.getLogger(__name__)


@dataclass
class RagasScores:
    faithfulness: float
    answer_relevance: float
    context_precision: float


def evaluate_with_ragas(
    question: str,
    answer: str,
    passages: list[RetrievedPassage],
    reference_answer: str | None = None,
    embedding_model: str = "dmis-lab/biobert-v1.1",
) -> RagasScores:
    """Compute hallucination metrics using embedding similarity (no OpenAI key needed).

    Faithfulness:       avg max-similarity between each answer sentence and passages.
    Answer Relevance:   cosine similarity between question and generated answer.
    Context Precision:  avg cosine similarity between question and each passage.
    """
    if not passages or not answer.strip():
        return RagasScores(faithfulness=0.0, answer_relevance=0.0, context_precision=0.0)

    model = SentenceTransformer(embedding_model)
    passage_texts = [p.text for p in passages]

    # Encode everything
    q_emb = model.encode([question], convert_to_tensor=True)
    a_emb = model.encode([answer], convert_to_tensor=True)
    p_embs = model.encode(passage_texts, convert_to_tensor=True)

    # --- Faithfulness: how well is the answer supported by passages? ---
    # Split answer into sentences, score each against best passage
    import re
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 10]
    if sentences:
        s_embs = model.encode(sentences, convert_to_tensor=True)
        sim_matrix = util.cos_sim(s_embs, p_embs)          # (n_sentences, n_passages)
        best_per_sentence = sim_matrix.max(dim=1).values    # best passage match per sentence
        faithfulness = float(best_per_sentence.mean().item())
    else:
        sim_ap = util.cos_sim(a_emb, p_embs)
        faithfulness = float(sim_ap.max().item())

    # --- Answer Relevance: is the answer relevant to the question? ---
    answer_relevance = float(util.cos_sim(q_emb, a_emb)[0][0].item())

    # --- Context Precision: are the retrieved passages relevant to the question? ---
    sim_qp = util.cos_sim(q_emb, p_embs)                   # (1, n_passages)
    context_precision = float(sim_qp.mean().item())

    # Clamp to [0, 1]
    faithfulness = max(0.0, min(1.0, faithfulness))
    answer_relevance = max(0.0, min(1.0, answer_relevance))
    context_precision = max(0.0, min(1.0, context_precision))

    return RagasScores(
        faithfulness=round(faithfulness, 4),
        answer_relevance=round(answer_relevance, 4),
        context_precision=round(context_precision, 4),
    )
