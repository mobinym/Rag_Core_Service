# app/strategies/vector_stores/impl.py
import os
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from .base import BaseVectorStoreStrategy

EMBEDDING_DIM = 1024

class FAISSStrategy(BaseVectorStoreStrategy):
    """پیاده‌سازی استراتژی برای FAISS با قابلیت ذخیره و بازیابی."""
    
    def __init__(self, embeddings=None):
        super().__init__(embeddings=FakeEmbeddings(size=EMBEDDING_DIM))

    def create_index(self, texts: List[str], vectors: List[List[float]], metadatas: List[dict]) -> None:
        text_embedding_pairs = list(zip(texts, vectors))
        self.vectorstore = FAISS.from_embeddings(
            text_embeddings=text_embedding_pairs,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        print("FAISS index created successfully in-memory.")
    
    def save_local(self, path: str) -> None:
        if not self.vectorstore:
            raise RuntimeError("Cannot save an uninitialized vector store.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.vectorstore.save_local(path)
        print(f"FAISS index saved to: {path}")

    def load_local(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No FAISS index found at {path}")
        self.vectorstore = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        print(f"FAISS index loaded from: {path}")

    def search(self, query_vector: List[float], k: int) -> List[Document]:
        if not self.vectorstore:
            raise RuntimeError("FAISS index is not initialized.")
            
        docs_with_scores = self.vectorstore.similarity_search_with_score_by_vector(query_vector, k=k)
        
        # ✅ --- راه حل نهایی اینجاست ---
        # در همین لحظه که امتیاز تولید می‌شود، آن را به float استاندارد پایتون تبدیل می‌کنیم.
        for doc, score in docs_with_scores:
            doc.metadata["score"] = float(score) 
        
        return [doc for doc, score in docs_with_scores]

# می‌توانید کلاس‌های ChromaStrategy و MilvusStrategy را نیز در آینده به همین شکل اصلاح کنید.