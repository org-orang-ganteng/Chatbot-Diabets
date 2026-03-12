# BioRAG Medical Assistant 🏥

نظام الكشف الآلي عن الهلوسة والتحقق من الحقائق في أنظمة الأسئلة والأجوبة الطبية باستخدام تقنية التوليد المسترجع المعزز (RAG)

## 📋 نظرة عامة

BioRAG هو نظام ذكاء اصطناعي متقدم للرعاية الصحية يعتمد على تقنية RAG (Retrieval-Augmented Generation) لضمان الدقة العلمية المطلقة. يقوم النظام بربط قدرات النماذج اللغوية الكبيرة بأحدث الأدبيات الطبية الموثقة من PubMed، مما يمنع تقديم معلومات خاطئة أو "مهلوسة" للمستخدمين.

## ✨ المميزات الرئيسية

- 🔍 **استرجاع ذكي**: البحث في قاعدة بيانات ضخمة من أبحاث PubMed
- 🤖 **توليد مدعوم بالأدلة**: إجابات مبنية على مصادر علمية موثقة
- ✅ **التحقق الآلي**: وحدة تحقق مبتكرة (Verification Module) تفحص كل جملة
- 📊 **درجة الموثوقية**: حساب Faithfulness Score لكل إجابة
- ⚠️ **تحذيرات الهلوسة**: تنبيه المستخدم عند اكتشاف معلومات غير مدعومة
- 📚 **مراجع علمية**: روابط مباشرة للأبحاث الأصلية

## 🏗️ البنية المعمارية

```
┌─────────────┐
│   السؤال    │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Embedding Model    │
│  (BGE-small)        │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Vector Database    │
│  (ChromaDB)         │
│  PubMed Data        │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  LLM Generator      │
│  (BioMistral-7B)    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Verification Module │ ← الابتكار الأساسي
│  (NLI DeBERTa)      │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Faithfulness Score │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  الإجابة النهائية  │
│  + المصادر          │
└─────────────────────┘
```

## 🚀 التثبيت والإعداد

### المتطلبات الأساسية

- Python 3.10+
- pip
- حساب Hugging Face (للحصول على API Token)

### خطوات التثبيت

1. **استنساخ المشروع**
```bash
git clone <repository-url>
cd BioRAG_Project
```

2. **إنشاء بيئة افتراضية**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. **تثبيت المكتبات**
```bash
pip install -r requirements.txt
```

4. **إعداد ملف البيئة**
```bash
# إنشاء ملف .env
echo HF_TOKEN=your_huggingface_token_here > .env
```

احصل على Token من: https://huggingface.co/settings/tokens

5. **تحميل بيانات PubMed**
```bash
python download_data.py
```

6. **بناء قاعدة البيانات المتجهة**
```bash
python fix_database.py
```

## 🎯 تشغيل التطبيق

```bash
python -m streamlit run app.py
```

سيفتح التطبيق تلقائياً في المتصفح على:
- Local URL: http://localhost:8501
- Network URL: http://192.168.100.115:8501

## 📁 هيكل المشروع

```
BioRAG_Project/
├── app.py                      # تطبيق Streamlit الرئيسي
├── config.py                   # إعدادات النظام
├── prompts.py                  # قوالب النصوص
├── requirements.txt            # المكتبات المطلوبة
├── download_data.py            # تحميل بيانات PubMed
├── setup_data.py               # بناء قاعدة البيانات
├── fix_database.py             # إصلاح قاعدة البيانات
│
├── services/                   # الخدمات الأساسية
│   ├── data_loader.py         # تحميل البيانات
│   ├── retriever.py           # البحث في قاعدة البيانات
│   ├── generator.py           # توليد الإجابات
│   ├── verifier.py            # التحقق من الإجابات
│   └── scorer.py              # حساب درجة الموثوقية
│
├── utils/                      # أدوات مساعدة
│   └── helpers.py
│
├── data/                       # البيانات
│   ├── local_diabetes_dataset/ # بيانات PubMed المحلية
│   └── raw_pdfs/              # ملفات PDF إضافية
│
└── vector_db/                  # قاعدة البيانات المتجهة
    └── chroma_store/
```

## 🔧 استكشاف الأخطاء

### مشكلة: ChromaDB Error - HNSW Index

**الأعراض:**
```
InternalError: Error loading hnsw index
```

**الحل:**
```bash
python fix_database.py
```

### مشكلة: HuggingFace API Error

**الأعراض:**
```
401 Unauthorized
```

**الحل:**
- تأكد من صحة HF_TOKEN في ملف .env
- تحقق من صلاحيات Token

### مشكلة: Out of Memory

**الحل:**
- قلل قيمة SAMPLE_SIZE في config.py
- استخدم نموذج embedding أصغر

## 📊 تقييم الأداء

النظام يستخدم إطار عمل RAGAS لتقييم:

- **Context Relevance**: مدى صلة المستندات المسترجعة
- **Answer Faithfulness**: مدى التزام الإجابة بالمصادر
- **Answer Relevance**: مدى صلة الإجابة بالسؤال

## 🔬 النماذج المستخدمة

| المكون | النموذج | الوظيفة |
|--------|---------|---------|
| Embedding | BAAI/bge-small-en-v1.5 | تحويل النصوص لمتجهات |
| LLM | BioMistral/BioMistral-7B | توليد الإجابات الطبية |
| NLI | cross-encoder/nli-deberta-v3-base | التحقق من الجمل |

## 🎓 الاستخدام الأكاديمي

إذا استخدمت هذا المشروع في بحثك، يرجى الإشارة إلى:

```bibtex
@software{biorag2024,
  title={BioRAG: Automated Hallucination Detection in Medical QA Systems},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## 📝 الترخيص

هذا المشروع مفتوح المصدر ومتاح للاستخدام الأكاديمي والبحثي.

## 🤝 المساهمة

نرحب بالمساهمات! يرجى:
1. Fork المشروع
2. إنشاء branch جديد
3. Commit التغييرات
4. Push إلى Branch
5. فتح Pull Request

## 📧 التواصل

للأسئلة والاستفسارات، يرجى فتح Issue في المستودع.

## 🙏 شكر وتقدير

- **PubMedQA Dataset**: qiaojin/PubMedQA
- **BioMistral**: نموذج LLM متخصص في الطب
- **LangChain**: إطار عمل RAG
- **ChromaDB**: قاعدة البيانات المتجهة

---

**⚠️ تنويه طبي مهم:**
هذا النظام أداة مساعدة للبحث والتعليم فقط. لا يجب استخدامه كبديل للاستشارة الطبية المهنية. استشر دائماً طبيباً مؤهلاً للحصول على المشورة الطبية.
