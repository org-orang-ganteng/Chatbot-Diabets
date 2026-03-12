from config import FAITHFULNESS_THRESHOLD

class BioScorer:
    def __init__(self):
        self.threshold = FAITHFULNESS_THRESHOLD

    def calculate_faithfulness(self, verification_results):
        """حساب نسبة صدق الإجابة بناءً على عدد الجمل المدعومة"""
        if not verification_results:
            return 0.0, False

        total_claims = len(verification_results)
        # نعد فقط الجمل التي حالتها SUPPORTED
        supported_claims = sum(1 for res in verification_results if res['status'] == 'SUPPORTED')
        
        # المعادلة: عدد الجمل الصحيحة / إجمالي الجمل
        score = supported_claims / total_claims
        
        # إذا كان السكور أعلى من الـ Threshold المحدد في config، فالإجابة موثوقة
        is_verified = score >= self.threshold
        
        return score, is_verified

# إنشاء نسخة جاهزة للاستخدام
scorer_service = BioScorer()