# rag_core_service/app/strategies/vector_stores/impl.py

import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_community.embeddings import FakeEmbeddings
from .base import BaseVectorStoreStrategy
import logging 
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss 
from langchain_community.retrievers import BM25Retriever

logger = logging.getLogger(__name__) 

EMBEDDING_DIM = 1024
#------------------------------------------------------------------------------------------------------------------------------------
class FAISSStrategy(BaseVectorStoreStrategy):

    def __init__(self, embeddings=None):
        super().__init__(embeddings=FakeEmbeddings(size=EMBEDDING_DIM))
        self.bm25_retriever: BM25Retriever = None 

    def _build_bm25_retriever(self):
        if not self.vectorstore or \
           not hasattr(self.vectorstore, 'docstore') or \
           not hasattr(self.vectorstore, 'index_to_docstore_id'):
            self.bm25_retriever = None
            return

        try:
            doc_ids = list(self.vectorstore.index_to_docstore_id.values())
            if not doc_ids:
                self.bm25_retriever = None
                return

            all_docs = [self.vectorstore.docstore.search(doc_id) for doc_id in doc_ids]
            valid_docs = [doc for doc in all_docs if doc is not None]

            if valid_docs:
                self.bm25_retriever = BM25Retriever.from_documents(valid_docs)
                # --- FIX: Attach to the vectorstore instance so Retriever can see it ---
                self.vectorstore.bm25_retriever = self.bm25_retriever
                logger.info(f"BM25Retriever built and attached to vectorstore ({len(valid_docs)} docs).")
            else:
                self.bm25_retriever = None

        except Exception as e:
            logger.error(f"BM25 build error: {e}")
            self.bm25_retriever = None

            
    def create_index(self, texts: List[str], vectors: List[List[float]], metadatas: List[dict]) -> None:
        if not texts or not vectors:
            raise ValueError("Texts and vectors cannot be empty for index creation.")
            
        text_embedding_pairs = list(zip(texts, vectors))
        self.vectorstore = FAISS.from_embeddings(
            text_embeddings=text_embedding_pairs,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        logger.info("FAISS index created successfully in-memory.")
        
        self._build_bm25_retriever() 

    def save_local(self, path: str) -> None:
        if not self.vectorstore:
            raise RuntimeError("Cannot save an uninitialized vector store. Call create_index first.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.vectorstore.save_local(path)
        logger.info(f"FAISS index saved to: {path}") 

    def load_local(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No FAISS index found at {path}")
            
        self.vectorstore = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        logger.info(f"FAISS index loaded from: {path}")
        
        self._build_bm25_retriever()

    def add_documents(self, texts: List[str], vectors: List[List[float]], metadatas: List[dict]):
        """اسناد جدید را به ایندکس FAISS موجود اضافه می‌کند."""
        if not self.vectorstore:
            raise RuntimeError("Vector store is not loaded or created yet.")
        text_embedding_pairs = list(zip(texts, vectors))
        self.vectorstore.add_embeddings(text_embeddings=text_embedding_pairs, metadatas=metadatas)
        logger.info(f"Added {len(texts)} new documents. Rebuilding BM25...")
        
        self._build_bm25_retriever() 

    def delete(self, ids: List[str]) -> bool:
        """
        وکتورها را بر اساس شناسه‌های داخلی FAISS حذف می‌کند.
        """
        if not self.vectorstore:
            return False
        try:
            self.vectorstore.delete(ids)
            logger.info(f"Successfully deleted {len(ids)} vectors from FAISS.")
            logger.info("Rebuilding BM25 after deletion...")
            
            self._build_bm25_retriever()
            return True
        except Exception as e:
            logger.error(f"Failed to delete from FAISS: {e}")
            return False
            
    def create_and_save_empty(self, path: str, metadatas: dict = None) -> None:
        """یک ایندکس خالی FAISS ساخته و ذخیره می‌کند."""
        empty_index = faiss.IndexFlatL2(EMBEDDING_DIM)
        empty_docstore = InMemoryDocstore({})
        empty_index_to_docstore_id = {}
        
        self.vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=empty_index,
            docstore=empty_docstore,
            index_to_docstore_id=empty_index_to_docstore_id
        )
        
        self._build_bm25_retriever() 
        
        self.save_local(path)
        logger.info(f"ایندکس خالی FAISS در مسیر زیر ساخته و ذخیره شد: {path}")
#------------------------------------------------------------------------------------------------------------------------------------
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
                self.vectorstore.persist() 
                self.vectorstore = None
                logger.info("Chroma vectorstore safely closed.")
            except Exception as e:
                logger.warning(f"Failed to close Chroma vectorstore cleanly: {e}")

    def create_and_save_empty(self, path: str, metadatas: dict = None) -> None:
        """یک ایندکس خالی ChromaDB ساخته و ذخیره می‌کند."""
        self.vectorstore = Chroma(
            persist_directory=path,
            embedding_function=self.embeddings,
            collection_metadata=metadatas
        )
        logger.info(f"ایندکس خالی ChromaDB در مسیر زیر ساخته و ذخیره شد: {path}")
