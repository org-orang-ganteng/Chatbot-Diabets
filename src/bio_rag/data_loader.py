from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterable

from datasets import Dataset, DatasetDict, load_dataset

from .config import DIABETES_KEYWORDS

logger = logging.getLogger(__name__)


@dataclass
class PubMedQASample:
    qid: str
    question: str
    context: str
    answer: str
    authors: str = ""
    year: str = ""
    journal: str = ""
    title: str = ""


def _normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def _extract_context_text(record: dict[str, Any]) -> str:
    context = record.get("context", "")

    if isinstance(context, dict):
        blocks = []
        for key in ("contexts", "sentences", "text", "abstract"):
            val = context.get(key)
            if isinstance(val, list):
                blocks.extend(str(v) for v in val)
            elif isinstance(val, str):
                blocks.append(val)
        if blocks:
            return _normalize_text(" ".join(blocks))

    if isinstance(context, list):
        return _normalize_text(" ".join(str(v) for v in context))

    if isinstance(context, str):
        return _normalize_text(context)

    long_answer = record.get("long_answer") or record.get("final_decision") or ""
    return _normalize_text(str(long_answer))


def _extract_answer_text(record: dict[str, Any]) -> str:
    for key in ("long_answer", "final_decision", "answer"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return _normalize_text(val)
    return ""


def _is_diabetes_related(question: str, context: str, keywords: Iterable[str]) -> bool:
    corpus = f"{question} {context}".lower()
    return any(keyword.lower() in corpus for keyword in keywords)


def load_diabetes_pubmedqa(
    dataset_name: str,
    max_samples: int = 2000,
    keywords: Iterable[str] = DIABETES_KEYWORDS,
) -> list[PubMedQASample]:
    # Load from multiple PubMedQA configs to maximize diabetes coverage.
    # pqa_labeled has expert answers; pqa_artificial has auto-generated ones.
    seen_qids: set[str] = set()
    filtered: list[PubMedQASample] = []

    for config_name in ("pqa_labeled", "pqa_artificial"):
        try:
            raw = load_dataset(dataset_name, config_name, trust_remote_code=True)
        except Exception:
            continue
        split = _pick_split(raw)

        for idx, record in enumerate(split):
            question = _normalize_text(str(record.get("question", "")))
            context = _extract_context_text(record)

            if not question or not context:
                continue

            if not _is_diabetes_related(question, context, keywords):
                continue

            qid = str(record.get("pubid", f"{config_name}_{idx}"))
            if qid in seen_qids:
                continue
            seen_qids.add(qid)

            filtered.append(
                PubMedQASample(
                    qid=qid,
                    question=question,
                    context=context,
                    answer=_extract_answer_text(record),
                )
            )

            if len(filtered) >= max_samples:
                break

        if len(filtered) >= max_samples:
            break

    logger.info("Loaded %d diabetes-related samples from PubMedQA", len(filtered))

    # Fetch PubMed metadata (authors, year, journal) in batch
    _enrich_with_pubmed_metadata(filtered)

    return filtered


def _enrich_with_pubmed_metadata(samples: list[PubMedQASample]) -> None:
    """Fetch author/year/journal from PubMed API for all samples."""
    if not samples:
        return
    pubids = [s.qid for s in samples if s.qid.isdigit()]
    if not pubids:
        return
    metadata: dict[str, dict] = {}
    for i in range(0, len(pubids), 200):
        batch = pubids[i:i+200]
        ids_str = ",".join(batch)
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={ids_str}&retmode=json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "BioRAG/1.0"})
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read())
            result = data.get("result", {})
            for pid in batch:
                if pid in result and isinstance(result[pid], dict):
                    metadata[pid] = result[pid]
        except Exception as e:
            logger.warning("PubMed metadata fetch failed: %s", e)
    for s in samples:
        info = metadata.get(s.qid)
        if not info:
            continue
        authors_list = info.get("authors", [])
        if authors_list:
            names = [a.get("name", "") for a in authors_list[:3]]
            s.authors = ", ".join(names)
            if len(authors_list) > 3:
                s.authors += " et al."
        pubdate = info.get("pubdate", "")
        if pubdate:
            s.year = pubdate.split()[0] if pubdate.split() else pubdate[:4]
        s.journal = info.get("source", "")
        s.title = info.get("title", "")


def _pick_split(raw: DatasetDict | Dataset) -> Dataset:
    if isinstance(raw, Dataset):
        return raw

    for candidate in ("train", "pqa_labeled", "validation", "test"):
        if candidate in raw:
            return raw[candidate]

    first_key = next(iter(raw.keys()))
    return raw[first_key]
