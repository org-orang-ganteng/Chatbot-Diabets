# Bio-RAG: Automated Framework for Hallucination Quantification in Evidence-Based Medical QA

This prototype implements a diabetes-focused Retrieval Augmented Generation (RAG) pipeline over PubMedQA.

## Features

- Loads and filters `qiaojin/PubMedQA` for diabetes-related samples
- Builds a knowledge base from PubMed contexts
- Indexes embeddings in FAISS
- Retrieves top-k evidence passages for a user question
- Generates an evidence-grounded biomedical answer with citations
- Decomposes answer into atomic claims
- Verifies claims against retrieved evidence via semantic similarity
- Computes hallucination metrics with RAGAS
- Returns answer, evidence, and trust score

## Project Structure

- `main.py`: CLI runner
- `src/bio_rag/config.py`: environment-driven configuration
- `src/bio_rag/data_loader.py`: PubMedQA loading and diabetes filtering
- `src/bio_rag/knowledge_base.py`: FAISS knowledge base build/load
- `src/bio_rag/retriever.py`: top-k passage retrieval
- `src/bio_rag/generator.py`: biomedical answer generation
- `src/bio_rag/claim_decomposer.py`: answer-to-claims splitting
- `src/bio_rag/claim_verifier.py`: claim support scoring
- `src/bio_rag/evaluator.py`: RAGAS metrics
- `src/bio_rag/pipeline.py`: end-to-end orchestration

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure (Optional)

```bash
cp .env.example .env
```

Default generator attempts a biomedical model (`microsoft/BioGPT-Large`) and falls back to `google/flan-t5-base` if unavailable.

## Run

```bash
python main.py --question "Can vitamin D help reduce complications in diabetes?"
```

The output JSON contains:

- `answer`
- `evidence` (retrieved passages)
- `claims` and `claim_checks`
- `trust_score` (supported claims / total claims)
- `ragas` metrics:
  - `faithfulness`
  - `answer_relevance`
  - `context_precision`

## Notes

- This is a research prototype; results depend on model quality and hardware.
- Large biomedical models may require GPU memory. If loading fails, fallback generation is used.
- For stricter context precision, provide curated reference answers/contexts.
