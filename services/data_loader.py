import os
import sys

# 1. حل مشكلة التعرف على المكتبات (Imports) برمجياً
try:
    import fitz  # PyMuPDF
except ImportError:
    print("⚠️ PyMuPDF (fitz) not found. Please ensure it is installed.")

try:
    # التحديث الجديد لـ LangChain يفضل الاستدعاء من هذا المسار المنفصل
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        # مسار احتياطي في حال استخدام نسخة قديمة من LangChain
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        print("⚠️ Langchain Text Splitter not found. Run: py -3.10 -m pip install langchain-text-splitters")

try:
    from datasets import load_dataset
except ImportError:
    print("⚠️ Datasets library not found. Run: py -3.10 -m pip install datasets")

# استيراد الإعدادات من ملف الإعدادات المركزي
try:
    from config import PDF_DIR
except ImportError:
    # مسار احتياطي في حال وجود مشكلة في استيراد config
    PDF_DIR = os.path.join(os.getcwd(), "data", "raw_pdfs")

class BioDataLoader:
    def __init__(self):
        # إعداد مقص النصوص (Text Splitter)
        # نقوم بتقسيم النص لقطع صغيرة لسهولة البحث والحفظ في قاعدة البيانات المتجهة
        try:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,    # حجم القطعة 500 حرف
                chunk_overlap=50   # تداخل بسيط لمنع ضياع المعنى بين القطع المتتالية
            )
        except NameError:
            self.splitter = None
            print("❌ Critical: RecursiveCharacterTextSplitter could not be loaded.")

    def load_from_pdf(self):
        """استخراج النصوص من ملفات PDF في المجلد المخصص"""
        all_texts = []
        if not os.path.exists(PDF_DIR):
            os.makedirs(PDF_DIR, exist_ok=True)
            return []
            
        for file in os.listdir(PDF_DIR):
            if file.endswith(".pdf"):
                path = os.path.join(PDF_DIR, file)
                try:
                    # فتح ملف PDF واستخراج النص من كل صفحة
                    doc = fitz.open(path)
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    all_texts.append(text)
                except Exception as e:
                    print(f"❌ Error reading PDF {file}: {e}")
        return all_texts

    def fetch_pubmed_diabetes(self, limit=50):
        """جلب بيانات السكري من Dataset العالمية (PubMedQA)"""
        # تم تقليل الـ limit الافتراضي قليلاً لضمان سرعة التحميل في المرة الأولى
        print("📥 Fetching Diabetes data from PubMedQA...")
        try:
            # تحميل البيانات مع تفعيل trust_remote_code لتفادي مشاكل الأمان في النسخ الجديدة
            dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train", trust_remote_code=True)
            
            # فلترة الأبحاث التي تتعلق بمرض السكري فقط
            diabetes_docs = [
                " ".join(item['context']['contexts']) 
                for item in dataset 
                if 'diabetes' in item['question'].lower()
            ]
            print(f"✅ Found {len(diabetes_docs)} relevant papers.")
            return diabetes_docs[:limit]
        except Exception as e:
            print(f"❌ Error fetching dataset: {e}")
            return []

    def process_and_split(self, docs):
        """تحويل النصوص الطويلة إلى قطع صغيرة (Chunks) جاهزة للفهرسة في قاعدة البيانات"""
        if self.splitter is None:
            print("❌ Cannot split documents: Splitter not initialized.")
            return []
        
        if not docs:
            return []
            
        # تحويل النصوص العادية إلى Document Objects يفهمها نظام ChromaDB
        return self.splitter.create_documents(docs)

# إنشاء نسخة جاهزة للاستخدام (Singleton Pattern) لتجنب إعادة التحميل المتكرر
data_loader_service = BioDataLoader()