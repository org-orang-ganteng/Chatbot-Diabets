# BioRAG Medical Assistant 🏥

Automated Hallucination Detection and Fact-Checking System for Medical Question-Answering using a Generate-then-Verify Architecture.

## 📋 Overview

BioRAG is an advanced healthcare AI system that uses a **Generate-then-Verify** architecture to ensure scientific accuracy. The system first generates answers freely using a language model, then retrieves relevant medical literature from PubMed and verifies the generated answer against the sources using Natural Language Inference (NLI). This approach detects and flags hallucinated or unsupported medical information.

All models run **fully locally** — no API keys required.

## ✨ Key Features

- 🤖 **Free Generation**: AI generates detailed medical answers from its own knowledge
- 🔍 **Smart Retrieval**: Semantic search across a large PubMed diabetes research database
- ✅ **Automated Verification**: NLI-based verification module checks every sentence against sources
- 📊 **Faithfulness Score**: Computes a hybrid faithfulness score for each answer
- ⚠️ **Hallucination Warnings**: Alerts users when information is unsupported by medical literature
- 📚 **Source References**: Direct links to original PubMed research papers

## 🏗️ Architecture

The system follows a **Generate-then-Verify** pipeline:

```
┌──────────────────┐
│   User Question  │
└────────┬─────────┘
         │
         ▼
┌──────────────────────┐
│  LLM Free Generation │  ← Step 1: Generate answer without context
│  (FLAN-T5-base)      │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Vector Database     │  ← Step 2: Retrieve relevant sources
│  (ChromaDB + BGE)    │
│  PubMedQA Data       │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Verification Module │  ← Step 3: Compare answer vs sources
│  NLI (DeBERTa-v3)    │
│  + Semantic Similarity│
│  + Entity Guards     │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Faithfulness Score  │
│  + Verified Answer   │
│  + Source Documents  │
└──────────────────────┘
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.10+
- pip

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/org-orang-ganteng/Chatbot-Diabets.git
cd BioRAG_Project
```

2. **Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download PubMed data**
```bash
python download_data.py
```

5. **Build the vector database**
```bash
python fix_database.py
```

## 🎯 Running the Application

```bash
python -m streamlit run app.py
```

The application will open automatically in your browser at:
- Local URL: http://localhost:8501

## 📁 Project Structure

```
BioRAG_Project/
├── app.py                      # Main Streamlit chatbot application
├── config.py                   # System configuration & constants
├── prompts.py                  # Prompt templates
├── requirements.txt            # Python dependencies
├── download_data.py            # PubMed data downloader
├── setup_data.py               # Database builder
├── fix_database.py             # Database repair utility
│
├── services/                   # Core services
│   ├── data_loader.py         # Data loading utilities
│   ├── retriever.py           # Vector database search
│   ├── generator.py           # Answer generation
│   ├── verifier.py            # Answer verification
│   └── scorer.py              # Faithfulness scoring
│
├── utils/                      # Helper utilities
│   └── helpers.py
│
├── assets/                     # Frontend assets
│   └── style.css              # Custom UI theme
│
├── data/                       # Data directory
│   ├── local_diabetes_dataset/ # Local PubMed diabetes data
│   └── raw_pdfs/              # Additional PDF files
│
└── vector_db/                  # Vector database
    └── chroma_store/          # ChromaDB persistent storage
```

## 🔧 Troubleshooting

### ChromaDB Error - HNSW Index

**Symptoms:**
```
InternalError: Error loading hnsw index
```

**Solution:**
```bash
python fix_database.py
```

### Out of Memory

**Solution:**
- Reduce `TOP_K_CANDIDATES` in `config.py`
- Use a smaller embedding model

## 🔬 Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Generator | google/flan-t5-base | Free medical answer generation |
| Embedding | BAAI/bge-small-en-v1.5 | Text-to-vector conversion for semantic search |
| NLI Verifier | cross-encoder/nli-deberta-v3-base | Sentence-level faithfulness verification |

## 📊 Verification Pipeline

The faithfulness scoring uses a **hybrid approach**:

1. **NLI Entailment (55%)**: Checks if source context entails each generated sentence
2. **Semantic Similarity (45%)**: Measures embedding similarity between claims and sources
3. **Entity Guard**: Penalizes scores when retrieved context doesn't cover question entities
4. **Coherence Check**: Ensures the answer is topically relevant to the question

Score thresholds:
- **≥ 70%**: ✅ Verified — Answer is supported by medical literature
- **40–69%**: ℹ️ Partial Match — Answer is partially supported
- **< 40%**: ⚠️ Low Match — Answer has low match with sources

## 🎓 Academic Use

If you use this project in your research, please cite:

```bibtex
@software{biorag2026,
  title={BioRAG: Automated Hallucination Detection in Medical QA using Generate-then-Verify},
  author={Aseel Flihan},
  year={2026},
  url={https://github.com/org-orang-ganteng/Chatbot-Diabets}
}
```

## 📝 License

This project is open-source and available for academic and research use.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📧 Contact

For questions and inquiries, please open an Issue in the repository.

## 🙏 Acknowledgments

- **PubMedQA Dataset**: qiaojin/PubMedQA
- **FLAN-T5**: Google's instruction-tuned text-to-text model
- **DeBERTa-v3**: Microsoft's NLI model for verification
- **LangChain**: RAG framework
- **ChromaDB**: Vector database

---

**⚠️ Medical Disclaimer:**
This system is a research and educational tool only. It should not be used as a substitute for professional medical advice. Always consult a qualified healthcare professional for medical guidance.
