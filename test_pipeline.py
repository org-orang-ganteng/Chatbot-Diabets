"""
Bio-RAG Step-by-Step Test
=========================
Runs each pipeline component individually so you can see intermediate results.

Stack:  BioMistral (LLM) | BioBERT (Embeddings) | FAISS (Retriever)

Usage:
    python test_pipeline.py
    python test_pipeline.py "Does metformin affect blood glucose levels?"
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict


# ── helpers ──────────────────────────────────────────────────────────────────

def banner(step: int, title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  STEP {step} -- {title}")
    print(f"{'='*70}\n", flush=True)


def elapsed(start: float) -> str:
    return f"{time.time() - start:.1f}s"


# ── test questions ───────────────────────────────────────────────────────────

TEST_QUESTIONS = [
    "Can vitamin D help reduce complications in diabetes?",
    "What is the relationship between insulin resistance and type 2 diabetes?",
    "Does metformin affect blood glucose levels in diabetic patients?",
]


def main() -> None:
    question = TEST_QUESTIONS[0]
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])

    print(f"Test question: {question}\n", flush=True)

    # ── Step 1: Load & filter dataset ────────────────────────────────────────
    banner(1, "Load PubMedQA & filter diabetes samples")
    t0 = time.time()

    from src.bio_rag.config import BioRAGConfig
    cfg = BioRAGConfig()
    print(f"  LLM          : {cfg.generator_model}")
    print(f"  Embedding    : {cfg.embedding_model}")
    print(f"  Retriever    : FAISS")

    from src.bio_rag.data_loader import load_diabetes_pubmedqa
    samples = load_diabetes_pubmedqa(
        dataset_name=cfg.dataset_name,
        max_samples=cfg.max_samples,
    )
    print(f"  Filtered samples : {len(samples)}  ({elapsed(t0)})")
    for s in samples[:3]:
        print(f"    [{s.qid}] {s.question[:90]}...")

    # ── Step 2: Build knowledge base (FAISS) ─────────────────────────────────
    banner(2, "Build / load FAISS knowledge base")
    t0 = time.time()

    from src.bio_rag.knowledge_base import KnowledgeBaseBuilder
    kb = KnowledgeBaseBuilder(cfg)
    vectorstore = kb.load_or_build(samples)
    print(f"  FAISS index ready ({elapsed(t0)})")

    # ── Step 3: Retrieve top-k passages ──────────────────────────────────────
    banner(3, f"Retrieve top-{cfg.top_k} passages for question")
    t0 = time.time()

    from src.bio_rag.retriever import BioRetriever
    retriever = BioRetriever(vectorstore, top_k=cfg.top_k)
    passages = retriever.retrieve(question)
    print(f"  Retrieved {len(passages)} passages ({elapsed(t0)})\n")

    for p in passages:
        snippet = p.text[:150].replace("\n", " ")
        print(f"  [Rank {p.rank}]  score={p.score:.4f}  qid={p.qid}")
        print(f"            {snippet}...\n")

    # ── Step 4: Generate answer with LLM ─────────────────────────────────────
    banner(4, "Generate answer with biomedical LLM")
    t0 = time.time()

    from src.bio_rag.generator import BiomedicalAnswerGenerator
    generator = BiomedicalAnswerGenerator(cfg.generator_model)
    answer = generator.generate(question, passages)
    print(f"  Generation time: {elapsed(t0)}")
    print(f"\n  ANSWER:\n  {answer}\n")

    # ── Step 5: Decompose answer into atomic claims ──────────────────────────
    banner(5, "Decompose answer into atomic claims")

    from src.bio_rag.claim_decomposer import decompose_into_claims
    claims = decompose_into_claims(answer)
    print(f"  Found {len(claims)} claim(s):\n")
    for i, c in enumerate(claims, 1):
        print(f"    {i}. {c}")

    # ── Step 6: Verify claims against retrieved context ──────────────────────
    banner(6, "Verify claims against evidence (cosine similarity)")
    t0 = time.time()

    from src.bio_rag.claim_verifier import ClaimVerifier, trust_score_from_claims
    verifier = ClaimVerifier(
        embedding_model=cfg.embedding_model,
        threshold=cfg.claim_similarity_threshold,
    )
    checks = verifier.verify(claims, passages)
    trust = trust_score_from_claims(checks)
    print(f"  Verification time: {elapsed(t0)}\n")

    for ch in checks:
        status = "SUPPORTED" if ch.supported else "NOT SUPPORTED"
        print(f"    [{status}] (score={ch.best_score:.3f}, evidence=E{ch.best_evidence_rank})")
        print(f'      "{ch.claim[:100]}"')

    print(f"\n  TRUST SCORE: {trust:.2%}")

    # ── Step 7: Hallucination metrics (embedding-based) ──────────────────────
    banner(7, "Hallucination metrics (BioBERT similarity)")
    t0 = time.time()

    from src.bio_rag.evaluator import evaluate_with_ragas
    ref = next((p.source_answer for p in passages if p.source_answer), None)
    ragas = evaluate_with_ragas(question, answer, passages, ref, embedding_model=cfg.embedding_model)
    print(f"  Evaluation time: {elapsed(t0)}")
    print(f"  Faithfulness       : {ragas.faithfulness}")
    print(f"  Answer Relevance   : {ragas.answer_relevance}")
    print(f"  Context Precision  : {ragas.context_precision}")

    # ── Step 8: Final output ─────────────────────────────────────────────────
    banner(8, "Final JSON output")
    output = {
        "question": question,
        "answer": answer,
        "evidence": [
            {"rank": p.rank, "score": p.score, "qid": p.qid, "text": p.text[:200] + "..."}
            for p in passages
        ],
        "claims": claims,
        "claim_checks": [asdict(ch) for ch in checks],
        "trust_score": trust,
        "ragas": asdict(ragas),
    }
    print(json.dumps(output, indent=2, default=str))
    print("\nPipeline test complete.")


if __name__ == "__main__":
    main()
