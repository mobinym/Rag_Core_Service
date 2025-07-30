# rag_core_service/app/strategies/vector_stores/base.py

from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

class BaseVectorStoreStrategy(ABC):
    def __init__(self, embeddings: Optional[Embeddings] = None):
        self.embeddings = embeddings
        self.vectorstore: Optional[Document] = None

    @abstractmethod
    def create_index(self, texts: List[str], vectors: List[List[float]], metadatas: List[dict]) -> None:
        pass

    @abstractmethod
    def save_local(self, path: str) -> None:
        pass

    @abstractmethod
    def load_local(self, path: str) -> None:
        pass
    
    @abstractmethod
    def create_and_save_empty(self, path: str, metadatas: dict = None) -> None:
        """یک ایندکس خالی ساخته و آن را در مسیر مشخص شده ذخیره می‌کند."""
        pass