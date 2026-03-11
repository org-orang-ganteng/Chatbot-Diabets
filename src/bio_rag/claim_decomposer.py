from __future__ import annotations

import re


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def decompose_into_claims(answer: str) -> list[str]:
    claims: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(answer.strip()):
        cleaned = sentence.strip(" -\n\t")
        if not cleaned:
            continue

        # Split on strong coordination to keep claims atomic.
        for chunk in re.split(r"\b(?:and|but|however|while)\b", cleaned, flags=re.IGNORECASE):
            claim = chunk.strip(" ,;")
            if len(claim) >= 20:
                claims.append(claim)

    return claims
