# app/strategies/vector_stores/base.py
from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

class BaseVectorStoreStrategy(ABC):
    def __init__(self, embeddings: Optional[Embeddings] = None):
        self.embeddings = embeddings
        self.vectorstore = None

    @abstractmethod
    def create_index(self, texts: List[str], vectors: List[List[float]], metadatas: List[dict]) -> None:
        pass

    @abstractmethod
    def search(self, query_vector: List[float], k: int) -> List[Document]:
        pass