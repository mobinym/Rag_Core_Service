# app/strategies/vector_stores/impl.py
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings # برای کار با بردارهای از پیش ساخته شده
from .base import BaseVectorStoreStrategy

# ابعاد مدل امبدینگ شما
EMBEDDING_DIM = 1024

class FAISSStrategy(BaseVectorStoreStrategy):
    """پیاده‌سازی استراتژی برای FAISS."""
    def __init__(self, embeddings=None): # این __init__ را اضافه می‌کنیم
        # ما اینجا embedder را نادیده می‌گیریم چون از بردارهای آماده استفاده می‌کنیم
        # اما برای هماهنگی با کلاس پایه، آن را تعریف می‌کنیم.
        super().__init__(embeddings=FakeEmbeddings(size=EMBEDDING_DIM))

    def create_index(self, texts: List[str], vectors: List[List[float]], metadatas: List[dict]) -> None:
        text_embedding_pairs = list(zip(texts, vectors))
        self.vectorstore = FAISS.from_embeddings(
            text_embeddings=text_embedding_pairs,
            embedding=self.embeddings, # از امبدینگ ساختگی استفاده می‌کنیم
            metadatas=metadatas
        )
        print("FAISS index created successfully in-memory.")

    def search(self, query_vector: List[float], k: int) -> List[Document]:
        if not self.vectorstore:
            raise RuntimeError("FAISS index is not initialized.")
        
        # جستجو بر اساس بردار و برگرداندن داکیومنت‌ها به همراه امتیاز
        docs_with_scores = self.vectorstore.similarity_search_with_score_by_vector(query_vector, k=k)
        # ما فقط داکیومنت‌ها را برمی‌گردانیم
        return [doc for doc, score in docs_with_scores]