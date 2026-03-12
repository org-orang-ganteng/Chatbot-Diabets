import os
from datasets import load_from_disk
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import *

LOCAL_DATA_PATH = os.path.join(DATA_DIR, "local_diabetes_dataset")
SAMPLE_SIZE = None  # None = process all records

def build_vector_db():
    # Verify local data exists
    if not os.path.exists(LOCAL_DATA_PATH):
        print(f"ERROR: Dataset not found at {LOCAL_DATA_PATH}")
        return

    print(f"Loading local dataset from: {LOCAL_DATA_PATH}")
    dataset = load_from_disk(LOCAL_DATA_PATH)
    print(f"Found {len(dataset)} rows.")

    sample = len(dataset) if SAMPLE_SIZE is None else min(SAMPLE_SIZE, len(dataset))
    print(f"Processing {sample} rows into documents...")

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

        # تخزين السؤال + السياق الطبي معاً لتحسين دقة البحث الدلالي
        if not context.strip():
            continue
        documents.append(Document(
            page_content=f"Question: {question}\n\nMedical Evidence: {context}",
            metadata={
                "source": "PubMedQA",
                "row_id": i,
                "question": question[:500],
                "answer": answer[:500]
            }
        ))

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    print(f"Embedding {len(chunks)} chunks and storing in ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_DIR)
    print("Vector database built successfully!")

if __name__ == "__main__":
    build_vector_db()