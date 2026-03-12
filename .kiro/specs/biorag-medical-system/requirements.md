# وثيقة المتطلبات - نظام BioRAG الطبي

## المقدمة

نظام BioRAG هو نظام ذكاء اصطناعي متقدم للرعاية الصحية يعتمد على تقنية التوليد المسترجع المعزز (Retrieval-Augmented Generation) لضمان الدقة العلمية المطلقة في الإجابة على الأسئلة الطبية. يقوم النظام بربط قدرات النماذج اللغوية الكبيرة بأحدث الأدبيات الطبية الموثقة من PubMed، مع وجود طبقة تحقق آلية تكشف "الهلوسة" (Hallucination) وتمنع تقديم معلومات خاطئة للمستخدمين.

## المصطلحات (Glossary)

- **BioRAG_System**: النظام الكامل المكون من خمس وحدات رئيسية (Data Loader, Retriever, Generator, Verifier, Scorer)
- **User**: المستخدم النهائي (طبيب، باحث، أو مريض) الذي يطرح أسئلة طبية
- **PubMed_Database**: قاعدة بيانات الأبحاث الطبية الموثقة من PubMedQA
- **Vector_Database**: قاعدة البيانات المتجهة (ChromaDB) التي تخزن النصوص الطبية بصيغة رياضية
- **Embedding_Model**: نموذج تحويل النصوص إلى متجهات (BAAI/bge-small-en-v1.5)
- **LLM_Generator**: النموذج اللغوي الكبير المتخصص في الطب (BioMistral-7B)
- **NLI_Model**: نموذج الاستدلال اللغوي الطبيعي (DeBERTa-v3) للتحقق من صحة الجمل
- **Faithfulness_Score**: درجة الموثوقية من 0 إلى 1 تقيس مدى التزام الإجابة بالمصادر
- **Hallucination**: معلومة خاطئة يولدها الذكاء الاصطناعي تبدو منطقية لكنها غير مدعومة بالمصادر
- **Context_Passages**: الفقرات الطبية المسترجعة من قاعدة البيانات
- **Atomic_Claims**: الجمل الصغيرة المستخرجة من الإجابة الطويلة للتحقق منها
- **Similarity_Threshold**: الحد الأدنى لدرجة التشابه (0.4) لقبول المستندات المسترجعة
- **Verification_Module**: وحدة التحقق المبتكرة (الصندوق الوردي) التي تفحص كل جملة

## المتطلبات

### المتطلب 1: تحميل وإدارة البيانات الطبية

**قصة المستخدم:** كمسؤول نظام، أريد تحميل وتخزين بيانات PubMed الخاصة بمرض السكري محلياً، حتى يتمكن النظام من العمل بدون الحاجة لإعادة التحميل في كل مرة.

#### معايير القبول

1. WHEN THE BioRAG_System receives a command to download data, THE BioRAG_System SHALL fetch the PubMedQA dataset from Hugging Face repository
2. WHILE THE BioRAG_System is filtering the dataset, THE BioRAG_System SHALL extract only records containing diabetes-related keywords in question, context, or answer fields
3. WHEN THE filtering process completes, THE BioRAG_System SHALL save the filtered dataset to local storage at data/local_diabetes_dataset directory
4. THE BioRAG_System SHALL download and cache NLTK punkt tokenizer data for sentence splitting operations
5. IF THE local dataset already exists, THEN THE BioRAG_System SHALL skip the download process and use the cached version

### المتطلب 2: بناء قاعدة البيانات المتجهة

**قصة المستخدم:** كمسؤول نظام، أريد تحويل البيانات الطبية النصية إلى قاعدة بيانات متجهة قابلة للبحث السريع، حتى يتمكن النظام من استرجاع المعلومات ذات الصلة بكفاءة.

#### معايير القبول

1. THE BioRAG_System SHALL load the Embedding_Model (BAAI/bge-small-en-v1.5) for text vectorization
2. WHEN THE BioRAG_System processes medical documents, THE BioRAG_System SHALL split each document into chunks of 500 characters with 50 characters overlap
3. THE BioRAG_System SHALL convert each text chunk into a numerical vector using the Embedding_Model
4. THE BioRAG_System SHALL store all vectors in the Vector_Database (ChromaDB) at vector_db/chroma_store directory
5. WHEN THE Vector_Database is corrupted, THE BioRAG_System SHALL provide a rebuild mechanism to recreate the database from source data

### المتطلب 3: استرجاع السياق الطبي ذي الصلة

**قصة المستخدم:** كمستخدم، أريد أن يبحث النظام في قاعدة البيانات الطبية عن معلومات ذات صلة بسؤالي، حتى تكون الإجابة مبنية على مصادر علمية موثقة.

#### معايير القبول

1. WHEN THE User submits a medical question, THE BioRAG_System SHALL convert the question into a vector using the Embedding_Model
2. THE BioRAG_System SHALL search the Vector_Database for the top 5 most similar Context_Passages using cosine similarity
3. IF THE similarity score of all retrieved passages is below Similarity_Threshold (0.4), THEN THE BioRAG_System SHALL return a message stating "No relevant medical information found"
4. WHEN THE BioRAG_System finds relevant passages, THE BioRAG_System SHALL concatenate them into a single context string
5. THE BioRAG_System SHALL preserve the source metadata for each retrieved passage for citation purposes

### المتطلب 4: توليد الإجابة الطبية

**قصة المستخدم:** كمستخدم، أريد الحصول على إجابة طبية دقيقة ومفصلة مبنية على المصادر المسترجعة، حتى أثق في المعلومات المقدمة.

#### معايير القبول

1. THE BioRAG_System SHALL use the LLM_Generator (BioMistral-7B) to generate medical answers
2. WHEN THE BioRAG_System generates an answer, THE BioRAG_System SHALL provide both the User question and the Context_Passages to the LLM_Generator
3. THE BioRAG_System SHALL configure the LLM_Generator with temperature 0.1 to minimize creative hallucination
4. THE BioRAG_System SHALL limit the generated answer to a maximum of 512 tokens
5. IF THE LLM_Generator API fails, THEN THE BioRAG_System SHALL return an error message to the User without crashing
6. THE BioRAG_System SHALL instruct the LLM_Generator to answer ONLY from the provided context and explicitly state when information is not available

### المتطلب 5: التحقق من صحة الإجابة (الوحدة المبتكرة)

**قصة المستخدم:** كمستخدم، أريد أن يتحقق النظام تلقائياً من صحة كل جملة في الإجابة مقابل المصادر الأصلية، حتى أتجنب المعلومات الخاطئة أو المهلوسة.

#### معايير القبول

1. WHEN THE BioRAG_System receives a generated answer, THE Verification_Module SHALL split the answer into Atomic_Claims using NLTK sentence tokenizer
2. THE Verification_Module SHALL filter out sentences shorter than 10 characters to avoid processing incomplete fragments
3. WHILE THE Verification_Module processes each claim, THE Verification_Module SHALL compare the claim against the Context_Passages using the NLI_Model
4. THE Verification_Module SHALL classify each claim as SUPPORTED, REFUTED, or NEUTRAL based on NLI_Model output
5. THE Verification_Module SHALL store the verification status and confidence score for each claim
6. THE Verification_Module SHALL return a structured list containing all claims with their verification results

### المتطلب 6: حساب درجة الموثوقية

**قصة المستخدم:** كمستخدم، أريد رؤية درجة رقمية توضح مدى موثوقية الإجابة، حتى أتخذ قراراً مستنيراً بشأن الاعتماد على المعلومات.

#### معايير القبول

1. THE BioRAG_System SHALL calculate the Faithfulness_Score as the ratio of SUPPORTED claims to total claims
2. WHEN THE Faithfulness_Score is greater than or equal to 0.7, THE BioRAG_System SHALL mark the answer as "Verified"
3. WHEN THE Faithfulness_Score is less than 0.7, THE BioRAG_System SHALL mark the answer with a "Hallucination Warning"
4. THE BioRAG_System SHALL display the Faithfulness_Score as a decimal number between 0.00 and 1.00
5. IF THE verification results list is empty, THEN THE BioRAG_System SHALL return a Faithfulness_Score of 0.0

### المتطلب 7: عرض النتائج والمصادر

**قصة المستخدم:** كمستخدم، أريد رؤية الإجابة مع درجة الموثوقية والمصادر الأصلية، حتى أتمكن من التحقق من المعلومات بنفسي.

#### معايير القبول

1. THE BioRAG_System SHALL display the generated answer in the main chat interface
2. WHEN THE answer is verified, THE BioRAG_System SHALL show a green success indicator with the text "Verified: Answer is supported by sources"
3. WHEN THE answer has low faithfulness, THE BioRAG_System SHALL show a yellow warning indicator with the text "Warning: Answer may not be fully supported"
4. THE BioRAG_System SHALL display the Faithfulness_Score as a metric with two decimal places
5. THE BioRAG_System SHALL provide an expandable section showing the original Context_Passages used for answer generation
6. THE BioRAG_System SHALL maintain a conversation history showing all previous questions and answers in the current session

### المتطلب 8: إدارة الأخطاء والحالات الاستثنائية

**قصة المستخدم:** كمستخدم، أريد أن يتعامل النظام بشكل صحيح مع الأخطاء والحالات غير المتوقعة، حتى لا يتعطل النظام أثناء الاستخدام.

#### معايير القبول

1. IF THE Vector_Database is corrupted or missing, THEN THE BioRAG_System SHALL display a clear error message with instructions to rebuild the database
2. IF THE Hugging Face API token is invalid or missing, THEN THE BioRAG_System SHALL return an authentication error message
3. WHEN THE LLM_Generator API times out, THE BioRAG_System SHALL wait up to 120 seconds before returning a timeout error
4. IF THE Embedding_Model fails to load, THEN THE BioRAG_System SHALL prevent the application from starting and display a model loading error
5. WHEN THE system encounters an out-of-memory error, THE BioRAG_System SHALL provide guidance to reduce the SAMPLE_SIZE configuration parameter

### المتطلب 9: الأداء والكفاءة

**قصة المستخدم:** كمستخدم، أريد أن يستجيب النظام بسرعة معقولة، حتى لا أنتظر وقتاً طويلاً للحصول على الإجابة.

#### معايير القبول

1. THE BioRAG_System SHALL cache the Embedding_Model in memory to avoid reloading on each query
2. THE BioRAG_System SHALL cache the NLI_Model in memory to avoid reloading during verification
3. THE BioRAG_System SHALL cache the Vector_Database connection to avoid reconnection overhead
4. WHEN THE BioRAG_System performs similarity search, THE BioRAG_System SHALL limit results to top 5 passages to balance relevance and speed
5. THE BioRAG_System SHALL process verification of multiple claims in sequence without blocking the user interface

### المتطلب 10: الأمان والخصوصية

**قصة المستخدم:** كمستخدم، أريد أن تكون بياناتي وأسئلتي محمية، حتى أشعر بالأمان عند استخدام النظام.

#### معايير القبول

1. THE BioRAG_System SHALL store the Hugging Face API token in a .env file that is excluded from version control
2. THE BioRAG_System SHALL not log or store User questions in persistent storage without explicit consent
3. THE BioRAG_System SHALL use HTTPS for all API communications with Hugging Face services
4. THE BioRAG_System SHALL validate and sanitize all User inputs before processing to prevent injection attacks
5. THE BioRAG_System SHALL display a medical disclaimer stating that the system is for research and educational purposes only
