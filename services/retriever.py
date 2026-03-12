from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import EMBEDDING_MODEL_NAME, CHROMA_DB_DIR, TOP_K_RETRIEVE

class BioRetriever:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vector_db = None

    def _get_db(self):
        if self.vector_db is None:
            self.vector_db = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=self.embeddings
            )
        return self.vector_db

    def search(self, query):
        db = self._get_db()
        docs = db.similarity_search(query, k=TOP_K_RETRIEVE)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context, docs

retriever_service = BioRetriever()