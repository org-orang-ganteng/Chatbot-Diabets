from __future__ import annotations

import logging
import re
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from .retriever import RetrievedPassage

logger = logging.getLogger(__name__)

GENERATOR_FALLBACK = "google/flan-t5-base"


class BiomedicalAnswerGenerator:
    """Generates answers using a biomedical LLM (BioMistral) with flan-t5 fallback."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._is_seq2seq = False
        self.tokenizer, self.model = self._load_model(model_name)

    def _load_model(self, model_name: str):
        # Try primary model (BioMistral / causal LM)
        # Skip large model download if insufficient disk space
        skip_primary = False
        try:
            import shutil
            free_gb = shutil.disk_usage("/").free / (1024 ** 3)
            if free_gb < 16 and "7b" in model_name.lower():
                logger.warning(
                    "Only %.1f GB free — skipping %s, using fallback %s",
                    free_gb, model_name, GENERATOR_FALLBACK,
                )
                skip_primary = True
        except Exception:
            pass

        if not skip_primary:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                # Try 4-bit quantization first (saves memory for 7B models)
                try:
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=bnb_config,
                        device_map="auto",
                    )
                    logger.info("Loaded %s in 4-bit quantization", model_name)
                    return tokenizer, model
                except Exception:
                    pass

                # Try float16 / float32 without quantization
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True,
                )
                logger.info("Loaded primary model: %s", model_name)
                return tokenizer, model

            except Exception as exc:
                logger.warning(
                    "Primary model %s failed (%s), falling back to %s",
                    model_name, exc, GENERATOR_FALLBACK,
                )

        # Fallback: flan-t5 (seq2seq)
        self._is_seq2seq = True
        tokenizer = AutoTokenizer.from_pretrained(GENERATOR_FALLBACK)
        model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_FALLBACK)
        logger.info("Loaded fallback seq2seq model: %s", GENERATOR_FALLBACK)
        return tokenizer, model

    def generate(self, question: str, passages: Iterable[RetrievedPassage]) -> str:
        passage_list = list(passages)

        # Strategy 1: Try model generation
        model_answer = self._try_model_generation(question, passage_list)

        # Strategy 2: Build answer from source Q&A pairs in evidence
        source_answer = _build_answer_from_sources(question, passage_list)

        # Combine: if model answer is good, use it as lead + enrich with sources
        if _is_good_answer(model_answer, question):
            if source_answer and len(source_answer) > len(model_answer):
                return f"{model_answer}\n\n{source_answer}"
            return model_answer

        # Model answer was weak, use source-based answer
        if source_answer:
            logger.info("Model answer too weak, using source-QA strategy")
            return source_answer

        # Last resort
        return model_answer if model_answer else "No relevant evidence found for this question."

    def _try_model_generation(self, question: str, passages: list[RetrievedPassage]) -> str:
        """Try generating with the LLM using an optimized prompt."""
        prompt = _format_prompt(question, passages, seq2seq=self._is_seq2seq)

        max_input = getattr(self.tokenizer, "model_max_length", 512)
        if max_input > 100_000:
            max_input = 512

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input,
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        if self._is_seq2seq:
            decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        else:
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return decoded.strip()


def _is_good_answer(answer: str, question: str) -> bool:
    """Check if an answer is meaningful and relevant to the question."""
    if not answer or len(answer) < 30:
        return False
    # Extract key nouns from question
    q_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', question.lower()))
    a_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', answer.lower()))
    # At least some question keywords should appear in the answer
    overlap = q_words & a_words
    if len(overlap) < 1:
        return False
    return True


def _format_prompt(question: str, passages: list[RetrievedPassage], *, seq2seq: bool = False) -> str:
    if seq2seq:
        # For seq2seq: include source Q&A pairs as examples + context
        qa_context_parts = []
        for p in passages[:3]:
            part = p.text[:350]
            if p.source_answer:
                part = f"Q: {p.source_question}\nA: {p.source_answer}\nContext: {p.text[:200]}"
            qa_context_parts.append(part)
        context = "\n\n".join(qa_context_parts)
        return (
            f"Based on the following medical research, answer the question in detail.\n\n"
            f"{context[:2000]}\n\n"
            f"Question: {question}\n"
            f"Detailed answer:"
        )

    evidence_block = "\n\n".join(
        [f"[E{p.rank}] {p.text[:500]}" for p in passages]
    )
    return (
        "You are a biomedical QA assistant. Use only the evidence passages below "
        "to answer the question. If evidence is insufficient, explicitly say so. "
        "Cite evidence as [E#].\n\n"
        f"Question: {question}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        "Answer:"
    )


def _build_answer_from_sources(question: str, passages: list[RetrievedPassage]) -> str:
    """Build a structured answer from evidence passages and their source Q&A metadata."""
    if not passages:
        return "No relevant evidence found for this question."

    q_lower = question.lower()

    # Collect relevant findings from source answers and context
    findings = []
    for p in passages[:5]:
        # Use the source answer (PubMedQA long_answer) if available and relevant
        if p.source_answer and len(p.source_answer) > 20:
            findings.append({
                "text": p.source_answer,
                "title": p.title,
                "year": p.year,
                "type": "answer",
            })
        else:
            # Extract the most relevant sentences from the evidence text
            sentences = re.split(r'(?<=[.!?])\s+', p.text)
            relevant = []
            for s in sentences:
                s = s.strip()
                if len(s) < 30:
                    continue
                # Prefer sentences with question keywords
                q_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', q_lower))
                s_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', s.lower()))
                if q_words & s_words:
                    relevant.append(s)
            if relevant:
                findings.append({
                    "text": " ".join(relevant[:2]),
                    "title": p.title,
                    "year": p.year,
                    "type": "context",
                })

    if not findings:
        # Final fallback: just use evidence text directly
        parts = []
        for p in passages[:3]:
            text = p.text[:300].strip()
            if len(text) > 300:
                text = text.rsplit(" ", 1)[0] + "…"
            parts.append(f"• {text}")
        return "Based on the available evidence:\n\n" + "\n\n".join(parts)

    # Build structured answer
    parts = []
    for f in findings[:3]:
        text = f["text"].strip()
        if len(text) > 500:
            text = text[:500].rsplit(" ", 1)[0] + "…"
        source_info = ""
        if f["title"]:
            source_info = f" (Source: {f['title']}"
            if f["year"]:
                source_info += f", {f['year']}"
            source_info += ")"
        parts.append(f"• {text}{source_info}")

    header = f"Based on the available research on \"{question}\":"
    return header + "\n\n" + "\n\n".join(parts)
