
import re

def clean_text(text):
    """تنظيف النص من المسافات الزائدة والرموز الغريبة"""
    if not text:
        return ""
    # إزالة المسافات المتكررة والأسطر الفارغة
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def format_claims_for_display(claims_list):
    """تنسيق قائمة الادعاءات الطبية لعرضها بشكل منظم"""
    formatted_text = ""
    for i, claim in enumerate(claims_list, 1):
        formatted_text += f"{i}. {claim}\n"
    return formatted_text

def calculate_percentage(score):
    """تحويل السكور العشري إلى نسبة مئوية للعرض"""
    return f"{score * 100:.1f}%"