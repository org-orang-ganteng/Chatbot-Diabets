import os
from datasets import load_dataset

# --- إعداد المسارات ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# المجلد الذي سنحفظ فيه البيانات بعد تحميلها وفلترتها
LOCAL_DATA_PATH = os.path.join(BASE_DIR, "data", "local_diabetes_dataset")

def setup_offline_data():
    print("⬇️ جاري تحميل مجموعة البيانات الأصلية من Hugging Face...")
    # بناءً على سجل الأوامر الخاص بك، يبدو أن اسم البيانات هو pqa_artificial
    dataset = load_dataset('qiaojin/PubMedQA', 'pqa_artificial', split='train')
    
    print(f"📊 عدد الصفوف الكلي قبل الفلترة: {len(dataset)}")

    print("🔍 جاري فلترة البيانات للبحث عن مرض السكري...")
    # دالة الفلترة (تبحث عن كلمة diabetes في السؤال أو السياق أو الإجابة)
    def filter_diabetes(example):
        text_to_search = f"{example.get('question', '')} {example.get('context', '')} {example.get('long_answer', '')}".lower()
        return 'diabetes' in text_to_search

    diabetes_dataset = dataset.filter(filter_diabetes)
    print(f"✅ عدد الصفوف بعد الفلترة (الخاصة بمرض السكري فقط): {len(diabetes_dataset)}")

    print(f"💾 جاري حفظ البيانات المفلترة محلياً في: {LOCAL_DATA_PATH}")
    # هذه الدالة ستقوم بحفظ البيانات بصيغة مُحسّنة (Arrow/Parquet) يمكن قراءتها لاحقاً بلمح البصر
    diabetes_dataset.save_to_disk(LOCAL_DATA_PATH)
    
    print("🎉 تم التنزيل والحفظ بنجاح! يمكنك الآن إغلاق هذا السكربت للأبد.")

if __name__ == "__main__":
    setup_offline_data()