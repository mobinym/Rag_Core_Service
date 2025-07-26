# rag_core_service/app/strategies/vector_stores/impl.py

import os
from typing import List
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import FakeEmbeddings
from .base import BaseVectorStoreStrategy

# ابعاد وکتورهای خروجی مدل امبدینگ شما
# این مقدار باید با خروجی سرویس embedding شما یکسان باشد
EMBEDDING_DIM = 1024

class FAISSStrategy(BaseVectorStoreStrategy):
    """پیاده‌سازی استراتژی Vector Store با استفاده از FAISS."""
    
    def __init__(self, embeddings=None):
        super().__init__(embeddings=FakeEmbeddings(size=EMBEDDING_DIM))

    def create_index(self, texts: List[str], vectors: List[List[float]], metadatas: List[dict]) -> None:
        if not texts or not vectors:
            raise ValueError("Texts and vectors cannot be empty for index creation.")
            
        text_embedding_pairs = list(zip(texts, vectors))
        self.vectorstore = FAISS.from_embeddings(
            text_embeddings=text_embedding_pairs,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        print("FAISS index created successfully in-memory.")
    
    def save_local(self, path: str) -> None:
        if not self.vectorstore:
            raise RuntimeError("Cannot save an uninitialized vector store. Call create_index first.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.vectorstore.save_local(path)
        print(f"FAISS index saved to: {path}")

    def load_local(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No FAISS index found at {path}")
            
        self.vectorstore = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        print(f"FAISS index loaded from: {path}")


class ChromaStrategy(BaseVectorStoreStrategy):
    """پیاده‌سازی استراتژی Vector Store با استفاده از ChromaDB. (نسخه نهایی)"""

    def __init__(self, embeddings=None):
        super().__init__(embeddings=FakeEmbeddings(size=EMBEDDING_DIM))
        self._temp_texts = None
        self._temp_vectors = None
        self._temp_metadatas = None

    def create_index(self, texts: List[str], vectors: List[List[float]], metadatas: List[dict]) -> None:
        """
        ✅ نهایی‌ترین اصلاح:
        یک ایندکس موقت در حافظه با استفاده از متد add_texts و پارامتر embeddings می‌سازد.
        """
        if not texts or not vectors:
            raise ValueError("Texts and vectors cannot be empty for index creation.")
        
        # ذخیره موقت داده‌ها برای استفاده در متد save_local
        self._temp_texts = texts
        self._temp_vectors = vectors
        self._temp_metadatas = metadatas

        # 1. یک نمونه خالی از Chroma می‌سازیم
        self.vectorstore = Chroma(embedding_function=self.embeddings)
        
        # 2. ✅ از متد add_texts همراه با پارامتر embeddings استفاده می‌کنیم
        self.vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            embeddings=vectors  # <--- این پارامتر کلیدی است
        )
        print("Chroma index created successfully in-memory.")

    def save_local(self, path: str) -> None:
        """
        ✅ نهایی‌ترین اصلاح:
        یک دیتابیس دائمی Chroma در مسیر داده‌شده ایجاد و داده‌ها را در آن ذخیره می‌کند.
        """
        if self._temp_texts is None:
            raise RuntimeError("Cannot save an uninitialized Chroma store. Call create_index first.")
        
        # 1. یک نمونه Chroma جدید با قابلیت ذخیره‌سازی دائمی در مسیر path می‌سازیم
        persistent_chroma = Chroma(
            persist_directory=path, 
            embedding_function=self.embeddings
        )

        # 2. ✅ از متد add_texts همراه با پارامتر embeddings برای افزودن داده‌ها استفاده می‌کنیم
        persistent_chroma.add_texts(
            texts=self._temp_texts,
            metadatas=self._temp_metadatas,
            embeddings=self._temp_vectors  # <--- این پارامتر کلیدی است
        )
        
        # در نسخه‌های جدید chromadb، نیازی به persist صریح نیست اما برای اطمینان نگه می‌داریم
        # persistent_chroma.persist()

        # 4. آبجکت vectorstore کلاس را به نسخه دائمی آپدیت می‌کنیم
        self.vectorstore = persistent_chroma
        print(f"ChromaDB created and persisted at: {path}")

    def load_local(self, path: str) -> None:
        """دیتابیس Chroma را از مسیر داده‌شده بارگذاری می‌کند."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No ChromaDB found at {path}")

        self.vectorstore = Chroma(persist_directory=path, embedding_function=self.embeddings)
        print(f"ChromaDB loaded from: {path}")