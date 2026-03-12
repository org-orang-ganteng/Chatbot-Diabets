"""Bio-RAG package for diabetes-focused hallucination quantification."""

__all__ = ["BioRAGPipeline"]


def __getattr__(name: str):
    if name == "BioRAGPipeline":
        from .pipeline import BioRAGPipeline
        return BioRAGPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
