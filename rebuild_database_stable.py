"""
إعادة بناء قاعدة البيانات باستخدام إصدار مستقر من ChromaDB
يحل مشكلة HNSW Index على Windows
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

def clean_everything():
    """حذف كامل لقاعدة البيانات"""
    print("🗑️  جاري حذف قاعدة البيانات القديمة بالكامل...")
    if os.path.exists(CHROMA_DB_DIR):
        try:
            # إغلاق أي عمليات مفتوحة
            import gc
            gc.collect()
            
            # حذف المجلد
            shutil.rmtree(CHROMA_DB_DIR, ignore_errors=True)
            print("✅ تم الحذف بنجاح")
        except Exception as e:
            print(f"⚠️  تحذير: {e}")
            print("ℹ️  سنحاول المتابعة...")
    
    # إعادة إنشاء المجلد
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    print("✅ تم إنشاء مجلد نظيف")

def build_with_stable_config():
    """بناء قاعدة البيانات بإعدادات مستقرة"""
    
    # التحقق من البيانات
    if not os.path.exists(LOCAL_DATA_PATH):
        print(f"❌ البيانات غير موجودة في {LOCAL_DATA_PATH}")
        return False

    print(f"\n📂 تحميل البيانات...")
    dataset = load_from_disk(LOCAL_DATA_PATH)
    print(f"✅ {len(dataset)} سجل")

    # معالجة البيانات
    sample = len(dataset) if SAMPLE_SIZE is None else min(SAMPLE_SIZE, len(dataset))
    print(f"\n📊 معالجة {sample} سجل...")

    documents = []
    for i in range(sample):
        row = dataset[i]
        question = row.get("question", "")
        answer = row.get("long_answer", "")
        
        context_data = row.get("context", {})
        if isinstance(context_data, dict) and "contexts" in context_data:
            context = " ".join(context_data["contexts"])
        else:
            context = str(context_data)

        # تخزين السياق الطبي فقط كمحتوى قابل للبحث
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

    print(f"✅ {len(documents)} مستند")

    # تقسيم النصوص
    print("\n✂️  تقسيم النصوص...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ {len(chunks)} قطعة")

    # تحميل Embeddings
    print(f"\n🔄 تحميل {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ تم التحميل")

    # بناء قاعدة البيانات بإعدادات محسّنة
    print(f"\n💾 بناء قاعدة البيانات...")
    print(f"📍 المسار: {CHROMA_DB_DIR}")
    print("⏳ قد يستغرق 2-3 دقائق...")
    
    try:
        # إنشاء قاعدة البيانات بدفعات صغيرة لتجنب مشاكل الذاكرة
        batch_size = 100
        vectordb = None
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            print(f"  📦 معالجة دفعة {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
            
            if vectordb is None:
                # إنشاء قاعدة البيانات في المرة الأولى
                vectordb = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory=CHROMA_DB_DIR,
                    collection_name="biorag_collection"
                )
            else:
                # إضافة للقاعدة الموجودة
                vectordb.add_documents(batch)
        
        print("✅ تم البناء بنجاح!")
        return True
        
    except Exception as e:
        print(f"❌ خطأ: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_search():
    """اختبار البحث"""
    print("\n🧪 اختبار البحث...")
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        
        db = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings,
            collection_name="biorag_collection"
        )
        
        # اختبار
        results = db.similarity_search("What are the symptoms of diabetes?", k=3)
        
        print(f"✅ نجح! وجدنا {len(results)} نتيجة")
        if results:
            print(f"\n📄 عينة: {results[0].page_content[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ فشل الاختبار: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("🔧 إعادة بناء قاعدة البيانات - نسخة مستقرة")
    print("=" * 70)
    
    # الخطوة 1: تنظيف كامل
    clean_everything()
    
    # الخطوة 2: البناء
    if not build_with_stable_config():
        print("\n❌ فشل البناء")
        exit(1)
    
    # الخطوة 3: الاختبار
    if not test_search():
        print("\n❌ فشل الاختبار")
        exit(1)
    
    print("\n" + "=" * 70)
    print("🎉 تم بنجاح! قاعدة البيانات جاهزة")
    print("=" * 70)
    print("\n✅ شغّل التطبيق: python -m streamlit run app.py")
