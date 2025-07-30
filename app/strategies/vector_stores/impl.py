# rag_core_service/app/strategies/vector_stores/impl.py

import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_community.embeddings import FakeEmbeddings
from .base import BaseVectorStoreStrategy
import logging # ✅ ایمپورت کردن logging
from langchain_community.docstore.in_memory import InMemoryDocstore
import numpy as np # ایمپورت جدید
import faiss # ایمپورت جدید

logger = logging.getLogger(__name__) 

EMBEDDING_DIM = 1024

class FAISSStrategy(BaseVectorStoreStrategy):

    
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
        # print("FAISS index created successfully in-memory.")
        logger.info("FAISS index created successfully in-memory.")

    def save_local(self, path: str) -> None:
        if not self.vectorstore:
            raise RuntimeError("Cannot save an uninitialized vector store. Call create_index first.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.vectorstore.save_local(path)
        # print(f"FAISS index saved to: {path}")
        logger.info(f"FAISS index saved to: {path}") 

    def load_local(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No FAISS index found at {path}")
            
        self.vectorstore = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        # print(f"FAISS index loaded from: {path}")
        logger.info(f"FAISS index loaded from: {path}") 

    def add_documents(self, texts: List[str], vectors: List[List[float]], metadatas: List[dict]):
        """اسناد جدید را به ایندکس FAISS موجود اضافه می‌کند."""
        if not self.vectorstore:
            raise RuntimeError("Vector store is not loaded or created yet.")
        text_embedding_pairs = list(zip(texts, vectors))
        self.vectorstore.add_embeddings(text_embeddings=text_embedding_pairs, metadatas=metadatas)
        # print(f"Added {len(texts)} new documents to the existing FAISS index.")
    def delete(self, ids: List[str]) -> bool:
        """
        وکتورها را بر اساس شناسه‌های داخلی FAISS حذف می‌کند.
        این متد توسط تابع index در LangChain استفاده می‌شود.
        """
        if not self.vectorstore:
            return False
        try:
            # LangChain FAISS wrapper handles the string-to-int conversion for IDs
            self.vectorstore.delete(ids)
            logger.info(f"Successfully deleted {len(ids)} vectors from FAISS.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete from FAISS: {e}")
            return False
    def create_and_save_empty(self, path: str, metadatas: dict = None) -> None:
        """یک ایندکس خالی FAISS ساخته و ذخیره می‌کند."""
        # ما باید یک ایندکس با ابعاد صحیح اما بدون داده بسازیم
        empty_index = faiss.IndexFlatL2(EMBEDDING_DIM)
        empty_docstore = InMemoryDocstore({})
        empty_index_to_docstore_id = {}
        
        # ساخت آبجکت LangChain FAISS از اجزای خالی
        self.vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=empty_index,
            docstore=empty_docstore,
            index_to_docstore_id=empty_index_to_docstore_id
        )
        self.save_local(path)
        logger.info(f"ایندکس خالی FAISS در مسیر زیر ساخته و ذخیره شد: {path}")

class ChromaStrategy(BaseVectorStoreStrategy):


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

        collection_metadata = {"hnsw:space": "cosine"}

        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            collection_metadata=collection_metadata 
        )
        
        self.vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            embeddings=vectors 
        )
        # print("Chroma index created successfully in-memory with COSINE distance.")
        logger.info("Chroma index created successfully in-memory with COSINE distance.")

    def save_local(self, path: str) -> None:
        if self._temp_texts is None:
            raise RuntimeError("Cannot save an uninitialized Chroma store. Call create_index first.")
        
        collection_metadata = {"hnsw:space": "cosine"}
        
        persistent_chroma = Chroma(
            persist_directory=path, 
            embedding_function=self.embeddings,
            collection_metadata=collection_metadata 
        )

        persistent_chroma.add_texts(
            texts=self._temp_texts,
            metadatas=self._temp_metadatas,
            embeddings=self._temp_vectors  
        )
        
        self.vectorstore = persistent_chroma
        # print(f"ChromaDB created and persisted at: {path} with COSINE distance.")
        logger.info(f"ChromaDB created and persisted at: {path} with COSINE distance.") 

    def load_local(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No ChromaDB found at {path}")

        self.vectorstore = Chroma(persist_directory=path, embedding_function=self.embeddings)
        # print(f"ChromaDB loaded from: {path}")
        logger.info(f"ChromaDB loaded from: {path}") 

    def add_documents(self, texts: List[str], vectors: List[List[float]], metadatas: List[dict]):
        """اسناد جدید را به ایندکس Chroma موجود اضافه می‌کند."""
        if not self.vectorstore:
            raise RuntimeError("Vector store is not loaded or created yet.")
        self.vectorstore.add_texts(texts=texts, metadatas=metadatas, embeddings=vectors)
        # print(f"Added {len(texts)} new documents to the existing Chroma index.")
    def delete(self, doc_uuids: List[str]) -> bool:
        """چانک‌ها را بر اساس لیستی از doc_uuid ها حذف می‌کند."""
        if not self.vectorstore:
            return False
        try:
            # Chroma می‌تواند مستقیماً بر اساس فراداده حذف کند
            self.vectorstore.delete(where={"doc_uuid": {"$in": doc_uuids}})
            logger.info(f"Successfully deleted chunks for doc_uuids: {doc_uuids}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents from Chroma: {e}")
            return False
    def close(self):
        """به‌صورت ایمن اتصال به ChromaDB را می‌بندد (برای جلوگیری از خطای حذف فایل در ویندوز)."""
        if self.vectorstore and hasattr(self.vectorstore, 'persist'):
            try:
                self.vectorstore.persist()  # اگر چیزی برای ذخیره باشه، ذخیره می‌کنه
                self.vectorstore = None
                logger.info("Chroma vectorstore safely closed.")
            except Exception as e:
                logger.warning(f"Failed to close Chroma vectorstore cleanly: {e}")

    def create_and_save_empty(self, path: str, metadatas: dict = None) -> None:
        """یک ایندکس خالی ChromaDB ساخته و ذخیره می‌کند."""
        # برای Chroma، ساخت یک ایندکس خالی به سادگی مقداردهی اولیه با یک مسیر است
        self.vectorstore = Chroma(
            persist_directory=path,
            embedding_function=self.embeddings,
            collection_metadata=metadatas
        )
        logger.info(f"ایندکس خالی ChromaDB در مسیر زیر ساخته و ذخیره شد: {path}")
