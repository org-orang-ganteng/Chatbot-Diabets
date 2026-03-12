"""
حل نهائي: استخدام FAISS بدلاً من ChromaDB
FAISS أكثر استقراراً على Windows ولا يعاني من مشاكل HNSW
"""
import os
import shutil
from datasets import load_from_disk
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import *

LOCAL_DATA_PATH = os.path.join(DATA_DIR, "local_diabetes_dataset")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_db")
SAMPLE_SIZE = None  # None = process all records

def clean_old_databases():
    """حذف قواعد البيانات القديمة"""
    print("🗑️  تنظيف قواعد البيانات القديمة...")
    
    # حذف ChromaDB
    chroma_path = os.path.join(BASE_DIR, "vector_db", "chroma_store")
    if os.path.exists(chroma_path):
        try:
            shutil.rmtree(chroma_path, ignore_errors=True)
            print("✅ تم حذف ChromaDB")
        except:
            pass
    
    # حذف FAISS القديم
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            shutil.rmtree(FAISS_INDEX_PATH, ignore_errors=True)
            print("✅ تم حذف FAISS القديم")
        except:
            pass
    
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    print("✅ جاهز للبناء")

def build_faiss_index():
    """بناء قاعدة بيانات FAISS"""
    
    if not os.path.exists(LOCAL_DATA_PATH):
        print(f"❌ البيانات غير موجودة: {LOCAL_DATA_PATH}")
        return False

    print(f"\n📂 تحميل البيانات...")
    dataset = load_from_disk(LOCAL_DATA_PATH)
    print(f"✅ {len(dataset)} سجل")

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

        full_text = f"Question: {question}\nContext: {context}\nAnswer: {answer}"
        documents.append(Document(
            page_content=full_text,
            metadata={"source": f"PubMed_{i}"}
        ))

    print(f"✅ {len(documents)} مستند")

    print("\n✂️  تقسيم النصوص...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ {len(chunks)} قطعة")

    print(f"\n🔄 تحميل نموذج التضمين...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    print("✅ تم التحميل")

    print(f"\n💾 بناء فهرس FAISS...")
    print("⏳ قد يستغرق دقيقتين...")
    
    try:
        vectordb = FAISS.from_documents(chunks, embeddings)
        vectordb.save_local(FAISS_INDEX_PATH)
        print("✅ تم البناء والحفظ بنجاح!")
        return True
    except Exception as e:
        print(f"❌ خطأ: {e}")
        return False

def test_faiss():
    """اختبار FAISS"""
    print("\n🧪 اختبار البحث...")
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        
        db = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        results = db.similarity_search("What are the symptoms of diabetes?", k=3)
        
        print(f"✅ نجح! {len(results)} نتيجة")
        if results:
            print(f"\n📄 عينة:\n{results[0].page_content[:250]}...")
        
        return True
    except Exception as e:
        print(f"❌ فشل: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("🔧 بناء قاعدة بيانات FAISS (حل نهائي)")
    print("=" * 70)
    
    clean_old_databases()
    
    if not build_faiss_index():
        print("\n❌ فشل البناء")
        exit(1)
    
    if not test_faiss():
        print("\n❌ فشل الاختبار")
        exit(1)
    
    print("\n" + "=" * 70)
    print("🎉 نجح! FAISS جاهز")
    print("=" * 70)
    print("\n⚠️  الآن عدّل app.py لاستخدام FAISS بدلاً من ChromaDB")
