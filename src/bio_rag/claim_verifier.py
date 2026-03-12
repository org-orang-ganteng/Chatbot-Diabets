from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers import SentenceTransformer, util

from .retriever import RetrievedPassage


@dataclass
class ClaimCheckResult:
    claim: str
    supported: bool
    best_score: float
    best_evidence_rank: int


class ClaimVerifier:
    def __init__(self, embedding_model: str, threshold: float = 0.62) -> None:
        self.model = SentenceTransformer(embedding_model)
        self.threshold = threshold

    def verify(self, claims: list[str], passages: list[RetrievedPassage]) -> list[ClaimCheckResult]:
        if not claims or not passages:
            return []

        claim_embeddings = self.model.encode(claims, convert_to_tensor=True)
        passage_embeddings = self.model.encode(
            [p.text for p in passages], convert_to_tensor=True
        )
        sim = util.cos_sim(claim_embeddings, passage_embeddings)

        results: list[ClaimCheckResult] = []
        for idx, claim in enumerate(claims):
            scores = sim[idx]
            best_pos = int(scores.argmax().item())
            best_score = float(scores[best_pos].item())
            evidence = passages[best_pos]

            results.append(
                ClaimCheckResult(
                    claim=claim,
                    supported=best_score >= self.threshold,
                    best_score=best_score,
                    best_evidence_rank=evidence.rank,
                )
            )

        return results


def trust_score_from_claims(results: list[ClaimCheckResult]) -> float:
    if not results:
        return 0.0
    supported = sum(1 for r in results if r.supported)
    return supported / len(results)
