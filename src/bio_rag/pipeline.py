from __future__ import annotations

from dataclasses import asdict, dataclass

from .claim_decomposer import decompose_into_claims
from .claim_verifier import ClaimVerifier, trust_score_from_claims
from .config import BioRAGConfig
from .data_loader import load_diabetes_pubmedqa
from .evaluator import evaluate_with_ragas
from .generator import BiomedicalAnswerGenerator
from .knowledge_base import KnowledgeBaseBuilder
from .retriever import BioRetriever, RetrievedPassage


@dataclass
class BioRAGResult:
    question: str
    answer: str
    evidence: list[RetrievedPassage]
    claims: list[str]
    claim_checks: list[dict]
    trust_score: float
    ragas: dict
    verified: bool = False
    hallucination_warning: bool = False
    no_evidence: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class BioRAGPipeline:
    def __init__(self, config: BioRAGConfig | None = None) -> None:
        self.config = config or BioRAGConfig()

        self.samples = load_diabetes_pubmedqa(
            dataset_name=self.config.dataset_name,
            max_samples=self.config.max_samples,
        )

        kb_builder = KnowledgeBaseBuilder(self.config)
        self.vectorstore = kb_builder.load_or_build(self.samples)

        self.retriever = BioRetriever(self.vectorstore, top_k=self.config.top_k)
        self.generator = BiomedicalAnswerGenerator(self.config.generator_model)
        self.verifier = ClaimVerifier(
            embedding_model=self.config.embedding_model,
            threshold=self.config.claim_similarity_threshold,
        )

    def ask(self, question: str) -> BioRAGResult:
        passages = self.retriever.retrieve(question)

        # Point 4: Check if documents are found
        if not passages:
            from .evaluator import RagasScores
            return BioRAGResult(
                question=question,
                answer="",
                evidence=[],
                claims=[],
                claim_checks=[],
                trust_score=0.0,
                ragas=asdict(RagasScores(0.0, 0.0, 0.0)),
                no_evidence=True,
            )

        answer = self.generator.generate(question, passages)

        claims = decompose_into_claims(answer)
        checks = self.verifier.verify(claims, passages)
        trust = trust_score_from_claims(checks)

        reference = self._best_reference_answer(passages)
        ragas_scores = evaluate_with_ragas(
            question, answer, passages, reference,
            embedding_model=self.config.embedding_model,
        )

        # Point 7: Check faithfulness threshold
        is_verified = ragas_scores.faithfulness > 0.7

        return BioRAGResult(
            question=question,
            answer=answer,
            evidence=passages,
            claims=claims,
            claim_checks=[asdict(c) for c in checks],
            trust_score=trust,
            ragas=asdict(ragas_scores),
            verified=is_verified,
            hallucination_warning=not is_verified,
        )

    def _best_reference_answer(self, passages: list[RetrievedPassage]) -> str | None:
        if not passages:
            return None

        for p in passages:
            if p.source_answer:
                return p.source_answer
        return None
