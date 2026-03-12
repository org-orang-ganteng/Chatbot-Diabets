"""
Bio-RAG Web Server
==================
Flask backend serving the landing page and chat API.
Supports Indonesian and English questions.
"""

from __future__ import annotations

import json
import logging
import re
import time
import threading

from flask import Flask, jsonify, request, send_from_directory
from transformers import MarianMTModel, MarianTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Indonesian → English translation map for medical terms ──────────────────
_ID_EN_MAP = {
    "apakah": "does", "apa": "what", "bagaimana": "how", "mengapa": "why",
    "bisakah": "can", "dapatkah": "can", "adakah": "is there",
    "diabetes": "diabetes", "diabetik": "diabetic", "gula darah": "blood glucose",
    "kadar gula": "blood sugar", "glukosa": "glucose", "insulin": "insulin",
    "resistensi insulin": "insulin resistance", "metformin": "metformin",
    "tekanan darah": "blood pressure", "kolesterol": "cholesterol",
    "obesitas": "obesity", "kegemukan": "obesity", "komplikasi": "complications",
    "pengobatan": "treatment", "terapi": "therapy", "obat": "medication",
    "penyakit jantung": "heart disease", "kardiovaskular": "cardiovascular",
    "pembuluh darah": "blood vessel", "ginjal": "kidney", "retinopati": "retinopathy",
    "neuropati": "neuropathy", "olahraga": "exercise", "latihan": "exercise",
    "pola makan": "diet", "makanan": "food", "nutrisi": "nutrition",
    "vitamin": "vitamin", "suplemen": "supplement",
    "pankreas": "pancreas", "transplantasi": "transplantation",
    "diagnosis": "diagnosis", "mendiagnosis": "diagnose",
    "gejala": "symptoms", "tanda": "signs",
    "pencegahan": "prevention", "mencegah": "prevent",
    "meningkatkan": "increase", "menurunkan": "reduce", "mempengaruhi": "affect",
    "risiko": "risk", "faktor risiko": "risk factor",
    "tipe 1": "type 1", "tipe 2": "type 2",
    "pasien": "patients", "penderita": "patients",
    "hubungan": "relationship", "pengaruh": "effect",
    "kontrol glikemik": "glycemic control", "HbA1c": "HbA1c",
    "menyebabkan": "cause", "mengakibatkan": "cause",
    "pada": "in", "dengan": "with", "untuk": "for",
    "dan": "and", "atau": "or", "dari": "from",
    "yang": "that", "ini": "this", "itu": "that",
    "dapat": "can", "bisa": "can", "mampu": "able to",
    "membantu": "help", "berpengaruh": "influential",
}

# ── Translation Model (EN → ID) ────────────────────────────────────────────
_translator_model = None
_translator_tokenizer = None
_translator_lock = threading.Lock()


def _load_translator():
    global _translator_model, _translator_tokenizer
    if _translator_model is not None:
        return
    with _translator_lock:
        if _translator_model is not None:
            return
        model_name = "Helsinki-NLP/opus-mt-en-id"
        logger.info("Loading translation model %s …", model_name)
        _translator_tokenizer = MarianTokenizer.from_pretrained(model_name)
        _translator_model = MarianMTModel.from_pretrained(model_name)
        logger.info("Translation model ready.")


def _translate_en_to_id(text: str) -> str:
    """Translate English text to Indonesian using MarianMT model."""
    _load_translator()

    # Split into sentences to translate individually for better quality
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]

    if not sentences:
        return text

    translated_parts = []
    # Translate in small batches
    batch_size = 8
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = _translator_tokenizer(batch, return_tensors="pt",
                                        padding=True, truncation=True,
                                        max_length=512)
        output_ids = _translator_model.generate(**inputs, max_new_tokens=512)
        for out in output_ids:
            decoded = _translator_tokenizer.decode(out, skip_special_tokens=True)
            translated_parts.append(decoded)

    result = " ".join(translated_parts)
    # Preserve bullet formatting
    result = result.replace("• ", "\n• ").strip()
    result = result.replace("\n\n• ", "\n• ")
    return result


def _is_indonesian(text: str) -> bool:
    """Heuristic: check if text contains common Indonesian words."""
    id_markers = {"apakah", "bagaimana", "mengapa", "bisakah", "dapatkah",
                  "adakah", "dengan", "pada", "untuk", "terhadap", "dari",
                  "gula darah", "kadar gula", "penderita", "penyakit",
                  "dan", "atau", "yang", "ini", "itu", "bisa", "dapat",
                  "obat", "menyebabkan", "berpengaruh", "membantu"}
    words = text.lower().split()
    matches = sum(1 for w in words if w in id_markers)
    return matches >= 2 or any(phrase in text.lower() for phrase in
                               ["gula darah", "kadar gula", "apakah", "bagaimana",
                                "bisakah", "dapatkah", "resistensi insulin",
                                "pola makan", "faktor risiko"])

def _translate_id_to_en(text: str) -> str:
    """Simple keyword-based Indonesian to English translation for medical queries."""
    result = text.lower()
    # Sort by length descending so longer phrases match first
    for id_term, en_term in sorted(_ID_EN_MAP.items(), key=lambda x: -len(x[0])):
        result = result.replace(id_term, en_term)
    # Clean up
    result = re.sub(r'\s+', ' ', result).strip()
    # Capitalize first letter
    if result:
        result = result[0].upper() + result[1:]
    return result

app = Flask(__name__, static_folder="static")

# Global pipeline instance (loaded once on first request)
_pipeline = None
_pipeline_lock = threading.Lock()
_pipeline_loading = False
_pipeline_error = None


def get_pipeline():
    global _pipeline, _pipeline_loading, _pipeline_error
    if _pipeline is not None:
        return _pipeline

    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        _pipeline_loading = True
        try:
            logger.info("Loading Bio-RAG pipeline (first request)...")
            from src.bio_rag.pipeline import BioRAGPipeline
            _pipeline = BioRAGPipeline()
            _pipeline_loading = False
            logger.info("Pipeline ready!")
            return _pipeline
        except Exception as e:
            _pipeline_loading = False
            _pipeline_error = str(e)
            logger.error("Pipeline load failed: %s", e)
            raise


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "ready": _pipeline is not None,
        "loading": _pipeline_loading,
        "error": _pipeline_error,
    })


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question is required"}), 400

    if len(question) > 500:
        return jsonify({"error": "Question too long (max 500 chars)"}), 400

    try:
        t0 = time.time()
        pipe = get_pipeline()

        # Detect Indonesian and translate for retrieval
        original_question = question
        is_id = _is_indonesian(question)
        if is_id:
            question_en = _translate_id_to_en(question)
            logger.info("ID→EN: '%s' → '%s'", question, question_en)
        else:
            question_en = question

        result = pipe.ask(question_en)
        duration = time.time() - t0

        # Translate answer back to Indonesian if question was in Indonesian
        answer_text = result.answer
        if is_id:
            answer_text = _translate_en_to_id(answer_text)

        evidence_list = []
        for p in result.evidence:
            evidence_list.append({
                "rank": p.rank,
                "score": round(p.score, 4),
                "qid": p.qid,
                "text": p.text[:300],
                "authors": p.authors,
                "year": p.year,
                "journal": p.journal,
                "title": p.title,
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{p.qid}/" if p.qid.isdigit() else "",
            })

        return jsonify({
            "question": original_question,
            "question_en": question_en if is_id else None,
            "language": "id" if is_id else "en",
            "answer": answer_text,
            "evidence": evidence_list,
            "claims": result.claims,
            "claim_checks": result.claim_checks,
            "trust_score": round(result.trust_score, 4),
            "ragas": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in result.ragas.items()
            },
            "duration": round(duration, 2),
        })

    except Exception as e:
        logger.exception("Error processing question")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
