# تعليمات نقل المشروع - CRITICAL FIX

## المشكلة الجذرية:
المسار الحالي يحتوي على أحرف عربية:
```
D:\s2\mata kulih\s2\حقي\BAHASA ALAMI\pak abadi\code\BioRAG_Project
```

هذا يسبب مشاكل مع:
- FAISS (لا يدعم Unicode paths)
- ChromaDB (مشاكل في HNSW index)
- العديد من المكتبات الأخرى

## الحل النهائي:

### الخطوة 1: انسخ المشروع لمسار إنجليزي
```cmd
xcopy "D:\s2\mata kulih\s2\حقي\BAHASA ALAMI\pak abadi\code\BioRAG_Project" "C:\Projects\BioRAG" /E /I /H
```

أو يدوياً:
1. افتح File Explorer
2. انسخ مجلد BioRAG_Project
3. الصقه في مسار إنجليزي مثل: `C:\Projects\BioRAG`

### الخطوة 2: افتح المشروع الجديد
```cmd
cd C:\Projects\BioRAG
```

### الخطوة 3: أعد تفعيل البيئة الافتراضية
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### الخطوة 4: شغّل سكريبت البناء
```cmd
python fix_with_faiss.py
```

### الخطوة 5: شغّل التطبيق
```cmd
python -m streamlit run app.py
```

## ملاحظة مهمة:
بعد النقل، ستحتاج لإعادة تحميل بيانات PubMed:
```cmd
python download_data.py
python fix_with_faiss.py
```

---

## البديل السريع (إذا لم تستطع نقل المشروع):

سأقوم بتعديل الكود ليحفظ قاعدة البيانات في مسار مؤقت بدون أحرف عربية:

```python
import tempfile
FAISS_INDEX_PATH = os.path.join(tempfile.gettempdir(), "biorag_faiss")
```

هل تريد:
1. نقل المشروع لمسار إنجليزي (الحل الأفضل) ✅
2. استخدام المسار المؤقت (حل سريع) ⚡
