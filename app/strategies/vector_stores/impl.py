# rag_core_service/app/strategies/vector_stores/impl.py

import os
from typing import List
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import FakeEmbeddings
from .base import BaseVectorStoreStrategy


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
        if not texts or not vectors:
            raise ValueError("Texts and vectors cannot be empty for index creation.")
        

        self._temp_texts = texts
        self._temp_vectors = vectors
        self._temp_metadatas = metadatas


        self.vectorstore = Chroma(embedding_function=self.embeddings)
        
        self.vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            embeddings=vectors 
        )
        print("Chroma index created successfully in-memory.")

    def save_local(self, path: str) -> None:
        if self._temp_texts is None:
            raise RuntimeError("Cannot save an uninitialized Chroma store. Call create_index first.")
        
        persistent_chroma = Chroma(
            persist_directory=path, 
            embedding_function=self.embeddings
        )

        persistent_chroma.add_texts(
            texts=self._temp_texts,
            metadatas=self._temp_metadatas,
            embeddings=self._temp_vectors  
        )
        

        self.vectorstore = persistent_chroma
        print(f"ChromaDB created and persisted at: {path}")

    def load_local(self, path: str) -> None:
        """دیتابیس Chroma را از مسیر داده‌شده بارگذاری می‌کند."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No ChromaDB found at {path}")

        self.vectorstore = Chroma(persist_directory=path, embedding_function=self.embeddings)
        print(f"ChromaDB loaded from: {path}")