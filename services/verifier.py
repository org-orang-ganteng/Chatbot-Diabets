import re
from sentence_transformers import CrossEncoder
import numpy as np
from config import NLI_MODEL_NAME
from utils.helpers import clean_text


def split_sentences(text):
    """Split text into sentences using regex (no nltk dependency)"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

class BioVerifier:
    def __init__(self):
        # تحميل "القاضي الآلي" (NLI Model)
        print(f"⏳ Loading Verification Model: {NLI_MODEL_NAME}...")
        self.nli_model = CrossEncoder(NLI_MODEL_NAME)
        # التصنيفات التي يخرجها النموذج: 0=Refuted, 1=Supported, 2=Neutral
        self.labels = ['REFUTED', 'SUPPORTED', 'NEUTRAL']

    def decompose_answer(self, answer):
        """تفكيك الإجابة الطويلة إلى جمل بسيطة قابلة للفحص"""
        sentences = split_sentences(answer)
        return [clean_text(s) for s in sentences if len(s) > 10]

    def verify_claims(self, context, answer):
        """فحص كل جملة في الإجابة مقابل السياق الطبي المسترجع"""
        claims = self.decompose_answer(answer)
        if not claims:
            return []

        verification_results = []
        
        for claim in claims:
            # المقارنة: هل السياق يدعم هذا الادعاء؟
            # الترتيب: (سياق البحث، الجملة المراد فحصها)
            scores = self.nli_model.predict([(context, claim)])
            label_idx = np.argmax(scores[0])
            status = self.labels[label_idx]
            
            verification_results.append({
                "claim": claim,
                "status": status,
                "score": float(np.max(scores[0]))
            })

        return verification_results

# إنشاء نسخة جاهزة للاستخدام
verifier_service = BioVerifier()