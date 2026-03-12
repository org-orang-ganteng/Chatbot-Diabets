"""
Script to download all required models once before running the app.
All models are local - NO API key needed.
Run this script first: python download_models.py
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_NAME, NLI_MODEL_NAME, GENERATOR_MODEL_NAME


def download_all_models():
    """Download all 3 models to local cache"""
    print("=" * 60)
    print("🚀 BioRAG Models Downloader")
    print("=" * 60)
    print(f"\nThis will download 3 models (~1.2GB total):")
    print(f"  1. Embedding:  {EMBEDDING_MODEL_NAME}  (~33MB)")
    print(f"  2. Generator:  {GENERATOR_MODEL_NAME}  (~990MB)")
    print(f"  3. Verifier:   {NLI_MODEL_NAME}  (~184MB)")
    print("\n🔒 No API key needed - everything runs locally!")
    print("You only need to run this once!\n")

    # 1. Embedding Model
    print("📥 [1/3] Downloading Embedding Model...")
    try:
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print("✅ Embedding model ready!\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return False

    # 2. Generator Model (FLAN-T5-base)
    print("📥 [2/3] Downloading Generator Model (FLAN-T5-base)...")
    try:
        AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
        AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL_NAME)
        print("✅ Generator model ready!\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return False

    # 3. NLI Verification Model
    print("📥 [3/3] Downloading Verification Model (DeBERTa-v3)...")
    try:
        AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
        print("✅ Verification model ready!\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return False

    print("=" * 60)
    print("🎉 All models downloaded successfully!")
    print("=" * 60)
    print("\n✨ You can now run the app with:")
    print("   python -m streamlit run app.py\n")
    return True


if __name__ == "__main__":
    success = download_all_models()
    if not success:
        print("\n⚠️  Some models failed. Check your internet connection and try again.")
        exit(1)
