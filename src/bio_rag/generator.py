from __future__ import annotations

import logging
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from .retriever import RetrievedPassage

logger = logging.getLogger(__name__)

GENERATOR_FALLBACK = "google/flan-t5-small"


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
        prompt = _format_prompt(question, passage_list, seq2seq=self._is_seq2seq)

        # Truncate prompt to model's max length
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
                max_new_tokens=256,
                do_sample=False,
            )

        if self._is_seq2seq:
            decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        else:
            # For causal LM, strip the input prompt from the output
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return decoded.strip() if decoded.strip() else "No answer generated."


def _format_prompt(question: str, passages: list[RetrievedPassage], *, seq2seq: bool = False) -> str:
    evidence_block = "\n\n".join(
        [f"[E{p.rank}] {p.text[:500]}" for p in passages]
    )
    if seq2seq:
        # Simpler prompt for small seq2seq models like flan-t5-small
        return (
            f"Answer the following medical question based on the context.\n\n"
            f"Context: {evidence_block[:1500]}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
    return (
        "You are a biomedical QA assistant. Use only the evidence passages below "
        "to answer the question. If evidence is insufficient, explicitly say so. "
        "Cite evidence as [E#].\n\n"
        f"Question: {question}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        "Answer:"
    )
