"""
سكريبت إصلاح قاعدة البيانات المتجهة
يقوم بحذف قاعدة البيانات التالفة وإعادة بنائها من الصفر
"""
import os
import shutil
from datasets import load_from_disk
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import *

LOCAL_DATA_PATH = os.path.join(DATA_DIR, "local_diabetes_dataset")
SAMPLE_SIZE = None  # None = process all records

def clean_old_database():
    """حذف قاعدة البيانات التالفة"""
    print("🗑️  جاري حذف قاعدة البيانات التالفة...")
    if os.path.exists(CHROMA_DB_DIR):
        try:
            shutil.rmtree(CHROMA_DB_DIR)
            print("✅ تم حذف قاعدة البيانات القديمة بنجاح")
        except Exception as e:
            print(f"❌ خطأ في الحذف: {e}")
            return False
    else:
        print("ℹ️  لا توجد قاعدة بيانات قديمة")
    
    # إعادة إنشاء المجلد الفارغ
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    print("✅ تم إنشاء مجلد جديد لقاعدة البيانات")
    return True

def build_vector_db():
    """بناء قاعدة البيانات من الصفر"""
    # التحقق من وجود البيانات المحلية
    if not os.path.exists(LOCAL_DATA_PATH):
        print(f"❌ خطأ: البيانات غير موجودة في {LOCAL_DATA_PATH}")
        print("ℹ️  قم بتشغيل download_data.py أولاً")
        return False

    print(f"\n📂 جاري تحميل البيانات من: {LOCAL_DATA_PATH}")
    dataset = load_from_disk(LOCAL_DATA_PATH)
    print(f"✅ تم العثور على {len(dataset)} سجل")

    # تحديد عدد السجلات للمعالجة
    sample = len(dataset) if SAMPLE_SIZE is None else min(SAMPLE_SIZE, len(dataset))
    print(f"\n📊 جاري معالجة {sample} سجل وتحويلها إلى مستندات...")

    documents = []
    for i in range(sample):
        row = dataset[i]
        question = row.get("question", "")
        answer = row.get("long_answer", "")

        # استخراج السياق
        context_data = row.get("context", {})
        if isinstance(context_data, dict) and "contexts" in context_data:
            context = " ".join(context_data["contexts"])
        else:
            context = str(context_data)

        # تخزين السياق الطبي فقط كمحتوى قابل للبحث
        # السؤال والإجابة يُحفظان كبيانات وصفية فقط
        if not context.strip():
            continue
        documents.append(Document(
            page_content=context,
            metadata={
                "source": "PubMedQA",
                "row_id": i,
                "question": question[:500],
                "answer": answer[:500]
            }
        ))

    print(f"✅ تم إنشاء {len(documents)} مستند")

    # تقسيم النصوص إلى قطع صغيرة
    print("\n✂️  جاري تقسيم النصوص إلى قطع صغيرة...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ تم إنشاء {len(chunks)} قطعة نصية")

    # تحميل نموذج التضمين
    print(f"\n🔄 جاري تحميل نموذج التضمين: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print("✅ تم تحميل نموذج التضمين بنجاح")

    # بناء قاعدة البيانات المتجهة
    print(f"\n💾 جاري بناء قاعدة البيانات المتجهة في: {CHROMA_DB_DIR}")
    print("⏳ هذه العملية قد تستغرق عدة دقائق...")
    
    try:
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_DIR
        )
        print("✅ تم بناء قاعدة البيانات بنجاح!")
        return True
    except Exception as e:
        print(f"❌ خطأ في بناء قاعدة البيانات: {e}")
        return False

def test_database():
    """اختبار قاعدة البيانات للتأكد من عملها"""
    print("\n🧪 جاري اختبار قاعدة البيانات...")
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        
        # اختبار بحث بسيط
        test_query = "What are the symptoms of diabetes?"
        results = db.similarity_search(test_query, k=3)
        
        print(f"✅ تم العثور على {len(results)} نتيجة للاختبار")
        print("\n📄 عينة من النتائج:")
        for i, doc in enumerate(results[:2], 1):
            preview = doc.page_content[:150] + "..."
            print(f"\n{i}. {preview}")
        
        return True
    except Exception as e:
        print(f"❌ فشل الاختبار: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("🔧 سكريبت إصلاح قاعدة البيانات المتجهة")
    print("=" * 70)
    
    # الخطوة 1: حذف قاعدة البيانات التالفة
    if not clean_old_database():
        print("\n❌ فشل في حذف قاعدة البيانات القديمة")
        exit(1)
    
    # الخطوة 2: بناء قاعدة البيانات الجديدة
    if not build_vector_db():
        print("\n❌ فشل في بناء قاعدة البيانات الجديدة")
        exit(1)
    
    # الخطوة 3: اختبار قاعدة البيانات
    if not test_database():
        print("\n❌ فشل اختبار قاعدة البيانات")
        exit(1)
    
    print("\n" + "=" * 70)
    print("🎉 تم إصلاح قاعدة البيانات بنجاح!")
    print("=" * 70)
    print("\n✅ يمكنك الآن تشغيل التطبيق باستخدام:")
    print("   python.exe -m streamlit run app.py")
