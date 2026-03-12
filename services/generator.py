"""
Generator service - fully local using FLAN-T5 model (no API key needed)
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import GENERATOR_MODEL_NAME
from prompts import RAG_SYSTEM_PROMPT


class BioGenerator:
    def __init__(self):
        print(f"⏳ Loading generator model: {GENERATOR_MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL_NAME)
        print("✅ Generator model loaded!")

    def generate(self, question, context):
        """توليد الإجابة الطبية باستخدام النموذج المحلي"""
        prompt = RAG_SYSTEM_PROMPT.format(context=context[:2000], question=question)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()


# إنشاء نسخة جاهزة للاستخدام
generator_service = BioGenerator()