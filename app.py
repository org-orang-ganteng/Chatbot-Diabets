"""
BioRAG Medical Assistant - Fully Local (No API Key Required)
All models run on your machine.
"""
import re
import numpy as np
import streamlit as st
import torch
from config import *
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from prompts import RAG_SYSTEM_PROMPT, FREE_GENERATION_PROMPT


def split_sentences(text):
    """Split text into sentences using regex (no nltk dependency)"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


# --- Page Configuration ---
st.set_page_config(page_title="BioRAG Medical Assistant", page_icon="🏥", layout="wide")

# --- Load Custom CSS ---
import os
css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Cached loaders ---
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

@st.cache_resource(show_spinner=False)
def load_nli():
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
    return tokenizer, model

@st.cache_resource(show_spinner=False)
def load_generator():
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL_NAME)
    return tokenizer, model

@st.cache_resource(show_spinner=False)
def load_vector_db(_embeddings):
    return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=_embeddings)


def get_db():
    """Lazy-load embeddings + vector DB (cached after first call)"""
    embeddings = load_embeddings()
    return load_vector_db(embeddings)


# ============================================================
#  Question-Driven Retrieval with Semantic Reranking
# ============================================================
def retrieve_relevant_docs(question):
    """
    1. Broad vector search (TOP_K_CANDIDATES)
    2. Rerank by question-to-original-question semantic similarity
    3. Return (top_docs, best_similarity_score)
    """
    db = get_db()
    emb = load_embeddings()

    candidates = db.similarity_search(question, k=TOP_K_CANDIDATES)
    if not candidates:
        return [], 0.0

    # Collect original questions from metadata
    orig_questions = []
    for doc in candidates:
        oq = (doc.metadata or {}).get("question", "")
        orig_questions.append(oq if oq else doc.page_content[:200])

    # Batch embed user question + all original questions
    q_vec = np.array(emb.embed_query(question))
    oq_vecs = np.array(emb.embed_documents(orig_questions))

    # Cosine similarity between user question and each original question
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)
    oq_norms = oq_vecs / (np.linalg.norm(oq_vecs, axis=1, keepdims=True) + 1e-8)
    similarities = oq_norms @ q_norm

    # Sort by question-intent similarity (descending)
    ranked = sorted(range(len(candidates)), key=lambda i: similarities[i], reverse=True)
    top_indices = ranked[:TOP_K_RETRIEVE]
    best_score = float(similarities[top_indices[0]])
    return [candidates[i] for i in top_indices], best_score


# ============================================================
#  Hybrid Faithfulness Scorer (NLI + Semantic + Entity Guard)
# ============================================================
def _extract_key_terms(text):
    """Extract meaningful medical/scientific terms (2+ chars, lowered)"""
    text_lower = text.lower()
    # Remove common stop words and keep meaningful tokens
    stop = {
        'the','a','an','is','are','was','were','be','been','being','have','has',
        'had','do','does','did','will','would','shall','should','may','might',
        'can','could','of','in','to','for','with','on','at','by','from','as',
        'into','through','during','before','after','above','below','between',
        'out','up','down','about','this','that','these','those','it','its',
        'and','or','but','not','no','nor','so','if','then','than','too','very',
        'just','also','how','what','which','who','whom','why','where','when',
        'all','each','every','both','few','more','most','other','some','such',
        'only','own','same','than','they','their','them','there','here','use',
        'used','using','does','improve','increase','decrease','cause','effect',
        'help','function','level','rate','risk','associated','related','study',
        'patient','patients','group','result','results','found','shown','based',
        'suggest','compared','significant','however','therefore','conclusion',
    }
    words = re.findall(r'\b[a-z]{3,}\b', text_lower)
    return set(w for w in words if w not in stop)


def _question_context_relevance(question, context):
    """
    Check if the retrieved context actually covers the key entities
    in the question. Returns a penalty multiplier [0.1 .. 1.0].
    
    Logic: extract key terms from question, check what fraction
    appears in context. Low overlap → heavy penalty.
    """
    q_terms = _extract_key_terms(question)
    if not q_terms:
        return 1.0

    ctx_lower = context.lower()
    matched = sum(1 for t in q_terms if t in ctx_lower)
    coverage = matched / len(q_terms)

    # If less than 30% of question terms found in context → heavy penalty
    if coverage < 0.3:
        return 0.15
    elif coverage < 0.5:
        return 0.4
    elif coverage < 0.7:
        return 0.7
    return 1.0


def _answer_question_coherence(question, answer):
    """
    Check if the answer is actually about the same topic as the question
    using embedding similarity. Returns a penalty multiplier [0.1 .. 1.0].
    """
    emb = load_embeddings()
    q_vec = np.array(emb.embed_query(question))
    a_vec = np.array(emb.embed_query(answer))

    q_n = q_vec / (np.linalg.norm(q_vec) + 1e-8)
    a_n = a_vec / (np.linalg.norm(a_vec) + 1e-8)
    sim = float(np.dot(q_n, a_n))

    # If answer is semantically far from question → it's off-topic
    if sim < 0.3:
        return 0.15
    elif sim < 0.5:
        return 0.4
    elif sim < 0.65:
        return 0.7
    return 1.0


def check_faithfulness(context, answer, question=""):
    """
    Smooth hybrid scoring with entity-aware guards:
    1. NLI entailment (continuous) against overlapping context windows
    2. Semantic similarity between each claim and best context window
    3. Entity overlap penalty (question terms vs context)
    4. Answer-Question coherence penalty
    → Final smooth score from 0.0 to 1.0
    """
    nli_tokenizer, nli_model = load_nli()
    emb = load_embeddings()

    valid_sentences = [s for s in split_sentences(answer) if len(s.strip()) >= 10]
    if not valid_sentences:
        return 0.5

    # Overlapping context windows (400 chars, step 300)
    ctx_windows = []
    for i in range(0, len(context), 300):
        w = context[i:i + 400]
        if len(w.strip()) > 20:
            ctx_windows.append(w)
    ctx_windows = (ctx_windows or [context[:512]])[:6]

    # --- Part A: Semantic similarity (batch) ---
    sent_vecs = np.array(emb.embed_documents(valid_sentences))
    ctx_vecs = np.array(emb.embed_documents(ctx_windows))

    sent_norms = sent_vecs / (np.linalg.norm(sent_vecs, axis=1, keepdims=True) + 1e-8)
    ctx_norms = ctx_vecs / (np.linalg.norm(ctx_vecs, axis=1, keepdims=True) + 1e-8)
    sim_matrix = sent_norms @ ctx_norms.T
    best_sims = np.clip(sim_matrix.max(axis=1), 0.0, 1.0)

    # --- Part B: NLI entailment (batch) ---
    premises = []
    hypotheses = []
    pair_sent_idx = []
    for i, sent in enumerate(valid_sentences):
        for window in ctx_windows:
            premises.append(window)
            hypotheses.append(sent[:256])
            pair_sent_idx.append(i)

    batch_inputs = nli_tokenizer(
        premises, hypotheses,
        return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    with torch.no_grad():
        batch_logits = nli_model(**batch_inputs).logits
    batch_probs = torch.softmax(batch_logits, dim=1)

    best_ent = [0.0] * len(valid_sentences)
    for k, si in enumerate(pair_sent_idx):
        ent = batch_probs[k][1].item()
        if ent > best_ent[si]:
            best_ent[si] = ent

    # --- Part C: Hybrid per-sentence score ---
    scores = []
    for i in range(len(valid_sentences)):
        hybrid = 0.55 * best_ent[i] + 0.45 * float(best_sims[i])
        scores.append(hybrid)

    raw_score = float(np.mean(scores))

    # --- Part D: Entity-aware penalties ---
    if question:
        entity_penalty = _question_context_relevance(question, context)
        coherence_penalty = _answer_question_coherence(question, answer)
        raw_score *= min(entity_penalty, coherence_penalty)

    return max(0.0, min(1.0, raw_score))


# ============================================================
#  Answer Generation (Free - without context)
# ============================================================
def generate_free_answer(question):
    """Generate answer using FLAN-T5 from its own knowledge (NO context)"""
    gen_tokenizer, gen_model = load_generator()
    prompt = FREE_GENERATION_PROMPT.format(question=question)
    inputs = gen_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_length=512,
            min_length=50,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.5
        )
    return gen_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <div style="font-size: 2.5rem;">🏥</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: #1e293b; margin-top: 0.3rem;">BioRAG</div>
        <div style="font-size: 0.8rem; color: #64748b;">Medical Hallucination Detector</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="padding: 0.6rem 0;">
        <div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">Pipeline</div>
        <div style="color: #334155; font-size: 0.85rem; line-height: 2;">
            <span style="color: #2563eb;">①</span> Generate Freely<br>
            <span style="color: #0d9488;">②</span> Retrieve Sources<br>
            <span style="color: #d97706;">③</span> Verify & Score
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="padding: 0.4rem 0;">
        <div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">Models</div>
        <div style="color: #475569; font-size: 0.78rem; line-height: 1.9;">
            🔬 <span style="color: #7c3aed;">""" + GENERATOR_MODEL_NAME + """</span><br>
            🛡️ <span style="color: #059669;">""" + NLI_MODEL_NAME.split('/')[-1] + """</span><br>
            🔢 <span style="color: #2563eb;">""" + EMBEDDING_MODEL_NAME.split('/')[-1] + """</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Refresh Database"):
        load_vector_db.clear()
        st.rerun()

    st.markdown("---")
    st.markdown('<div style="text-align:center; color: #64748b; font-size: 0.75rem;">🔒 Fully local · No API keys</div>', unsafe_allow_html=True)

# --- Chat UI ---
st.markdown("""
<div style="padding: 0.5rem 0 0.3rem;">
    <h1 style="color: #1e293b; font-size: 1.6rem; margin-bottom: 0.2rem;">🏥 BioRAG Medical Assistant</h1>
    <p style="color: #64748b; font-size: 0.88rem; margin: 0;">AI generates answers freely, then verifies against PubMed sources</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "score" in msg and msg["score"] is not None:
            pct = msg["score"]
            if pct >= FAITHFULNESS_THRESHOLD:
                icon, label, color = "✅", "Verified", "#059669"
            elif pct >= 0.4:
                icon, label, color = "ℹ️", "Partial", "#2563eb"
            else:
                icon, label, color = "⚠️", "Low Match", "#dc2626"
            st.markdown(f'<div style="color: {color}; font-size: 0.82rem; margin-top: 0.3rem;">{icon} Faithfulness: {pct:.0%} — {label}</div>', unsafe_allow_html=True)

if prompt := st.chat_input("Ask a medical question about diabetes..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        score = None
        context_text = ""
        retrieved_docs = []

        # ===== Step 1: Generate answer FREELY (no context) =====
        with st.spinner("🤖 AI is generating answer from its own knowledge..."):
            reply = generate_free_answer(prompt)

        st.markdown(reply)

        # ===== Step 2: Retrieve sources AFTER generation =====
        with st.spinner("🔍 Searching medical literature to verify..."):
            retrieved_docs, relevance_score = retrieve_relevant_docs(prompt)
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        sources_relevant = bool(context_text.strip()) and relevance_score >= MIN_RELEVANCE_THRESHOLD

        # ===== Step 3: Compare AI answer vs sources =====
        if sources_relevant:
            with st.spinner("🛡️ Comparing AI answer against medical sources..."):
                score = check_faithfulness(context_text, reply, question=prompt)

            # Determine result based on score
            if score >= FAITHFULNESS_THRESHOLD:
                score_icon = "✅"
                score_label = "Verified"
                score_desc = "Answer is supported by medical literature"
            elif score >= 0.4:
                score_icon = "ℹ️"
                score_label = "Partial Match"
                score_desc = "Answer is partially supported by sources"
            else:
                score_icon = "⚠️"
                score_label = "Low Match"
                score_desc = "Answer has low match with medical sources"

            if score >= FAITHFULNESS_THRESHOLD:
                color, bg = "#059669", "#f0fdf4"
            elif score >= 0.4:
                color, bg = "#2563eb", "#eff6ff"
            else:
                color, bg = "#dc2626", "#fef2f2"

            st.markdown(f"""
            <div style="margin-top: 1rem; padding: 1rem 1.2rem; background: {bg}; border: 1px solid {color}22; border-left: 3px solid {color}; border-radius: 10px;">
                <div style="display: flex; align-items: center; gap: 0.8rem;">
                    <span style="font-size: 1.8rem; font-weight: 700; color: {color};">{score:.0%}</span>
                    <div>
                        <div style="color: #1e293b; font-weight: 600; font-size: 0.9rem;">{score_icon} {score_label}</div>
                        <div style="color: #64748b; font-size: 0.78rem;">{score_desc} · {len(retrieved_docs)} sources checked</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📖 View Source Documents"):
                for i, doc in enumerate(retrieved_docs, 1):
                    meta = doc.metadata or {}
                    source_name = meta.get('source', 'N/A')
                    st.markdown(f'<div style="color: #2563eb; font-size: 0.82rem; font-weight: 600;">Source {i} <span style="color: #64748b; font-weight: 400;">— {source_name}</span></div>', unsafe_allow_html=True)
                    if meta.get('question'):
                        st.caption(f"Original Q: {meta['question'][:200]}")
                    st.text_area(
                        label=f"source_{i}",
                        value=doc.page_content[:500],
                        height=90,
                        key=f"src_{i}_{len(st.session_state.messages)}",
                        label_visibility="collapsed"
                    )
        else:
            st.warning("🚫 No relevant sources found — cannot verify this answer against PubMedQA database.")

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "score": score
    })

# Footer
st.markdown('<div style="text-align:center; padding: 1.5rem 0 0.5rem; border-top: 1px solid #e2e8f0; margin-top: 1rem;"><span style="color: #94a3b8; font-size: 0.75rem;">⚠️ For research and educational purposes only. Consult a qualified healthcare professional.</span></div>', unsafe_allow_html=True)