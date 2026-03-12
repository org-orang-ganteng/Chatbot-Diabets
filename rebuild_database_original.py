"""
Rebuild database with ONLY original PubMedQA data (no manual additions)
"""
import os
import shutil
from datasets import load_from_disk
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import *

print("=" * 70)
print("Rebuilding Vector Database with ORIGINAL PubMedQA Data ONLY")
print("=" * 70)

# Delete existing database
if os.path.exists(CHROMA_DB_DIR):
    print(f"\n🗑️  Deleting existing database at {CHROMA_DB_DIR}...")
    shutil.rmtree(CHROMA_DB_DIR)
    print("✅ Deleted")

# Load original dataset
LOCAL_DATA_PATH = os.path.join(DATA_DIR, "local_diabetes_dataset")
print(f"\n📥 Loading original PubMedQA dataset from {LOCAL_DATA_PATH}...")
dataset = load_from_disk(LOCAL_DATA_PATH)
print(f"✅ Loaded {len(dataset)} records")

# Process documents
SAMPLE_SIZE = None  # None = process all records
sample = len(dataset) if SAMPLE_SIZE is None else min(SAMPLE_SIZE, len(dataset))
print(f"\n📝 Processing {sample} records into documents...")

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
            "pubid": row.get("pubid", f"record_{i}"),
            "question": question[:500],
            "answer": answer[:500]
        }
    ))

print(f"✅ Created {len(documents)} documents")

# Split into chunks
print("\n✂️  Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks")

# Create embeddings and database
print("\n🔢 Creating embeddings and building ChromaDB...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_DIR)

print(f"\n✅ Database rebuilt successfully!")
print(f"📊 Total chunks in database: {len(db.get()['ids'])}")
print("\n" + "=" * 70)
print("✨ Database now contains ONLY original PubMedQA research data")
print("=" * 70)
