"""
Download all models ONCE before running the app
Run this script first: python setup_models.py
"""
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

print("=" * 70)
print("🚀 BioRAG - Pre-downloading All Models")
print("=" * 70)
print("\n⏳ This will take a few minutes. You only need to run this ONCE!\n")

# Model names
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
QA_MODEL = "distilgpt2"

# 1. Download Embedding Model
print("📥 [1/3] Downloading Embedding Model (BGE-small - 33MB)...")
try:
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("✅ Embedding model downloaded!\n")
except Exception as e:
    print(f"❌ Error: {e}\n")

# 2. Download QA Model
print("📥 [2/3] Downloading QA Model (DistilGPT2 - 82MB)...")
try:
    tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
    model = AutoModelForCausalLM.from_pretrained(QA_MODEL)
    print("✅ QA model downloaded!\n")
except Exception as e:
    print(f"❌ Error: {e}\n")

# 3. Download NLI Model
print("📥 [3/3] Downloading Verification Model (DeBERTa-v3 - 184MB)...")
try:
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    print("✅ Verification model downloaded!\n")
except Exception as e:
    print(f"❌ Error: {e}\n")

print("=" * 70)
print("🎉 All models downloaded successfully!")
print("=" * 70)
print("\n✨ Now you can run the app:")
print("   python -m streamlit run app_ultra_light.py\n")
print("💡 The app will start instantly without any downloads!\n")
