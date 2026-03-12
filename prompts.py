# ==========================================
# 1. قالب التوليد الحر (بدون مصادر - من معرفة النموذج فقط)
# ==========================================
# هذا القالب يجعل النموذج يجيب من معرفته الداخلية بدون أي سياق خارجي
FREE_GENERATION_PROMPT = """You are an expert Medical AI Assistant specializing in diabetes and metabolic diseases.
Answer the following medical question using your medical knowledge.

IMPORTANT INSTRUCTIONS:
1. Provide a DETAILED answer with at least 3-5 sentences.
2. Include specific medical facts, mechanisms, and clinical details.
3. Mention relevant biological processes, risk factors, or treatments.
4. Use professional medical terminology.
5. Structure your answer clearly.

Question:
{question}

Detailed Medical Answer:"""

# ==========================================
# 2. قالب التوليد المدعم بالمصادر (RAG Prompt) - يُستخدم كمرجع فقط
# ==========================================
RAG_SYSTEM_PROMPT = """You are an expert Medical AI Assistant. 
Use the following pieces of retrieved medical context to answer the user's question.

STRICT INSTRUCTIONS:
1. Use ONLY the provided context to answer. 
2. If the answer is not in the context, clearly state: "The provided medical literature does not contain information to answer this question."
3. Do NOT include any outside knowledge or common sense facts not present in the context.
4. Keep the answer professional, concise, and structured.

Context:
{context}

Question: 
{question}

Answer:"""

# ==========================================
# 2. قالب تفكيك الجمل (Claim Decomposition)
# ==========================================
# يُستخدم لتحويل الإجابة الطويلة إلى نقاط صغيرة (Claims) ليسهل فحصها في الصندوق الوردي.
DECOMPOSITION_PROMPT = """Break down the following medical statement into a list of individual factual claims. 
Each claim must be a single, independent sentence.

Statement:
{answer}

Factual Claims:"""

# ==========================================
# 3. قالب محاكاة الهلوسة (للأغراض التعليمية/الاختبار)
# ==========================================
HALLUCINATION_TEST_PROMPT = "Generate a plausible-sounding but medically incorrect fact about insulin."