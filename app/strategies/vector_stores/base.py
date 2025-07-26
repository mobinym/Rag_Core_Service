# rag_core_service/app/strategies/vector_stores/base.py

from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

class BaseVectorStoreStrategy(ABC):
    """
    کلاس پایه و انتزاعی برای استراتژی‌های مختلف Vector Store.
    این کلاس تضمین می‌کند که تمام پیاده‌سازی‌ها، متدهای لازم برای ایجاد،
    ذخیره و بازیابی ایندکس را داشته باشند.
    """
    def __init__(self, embeddings: Optional[Embeddings] = None):
        """
        سازنده کلاس پایه.
        یک مدل embedding (معمولا FakeEmbeddings چون وکتورها از قبل آماده‌اند) را دریافت می‌کند.
        """
        self.embeddings = embeddings
        self.vectorstore: Optional[Document] = None

    @abstractmethod
    def create_index(self, texts: List[str], vectors: List[List[float]], metadatas: List[dict]) -> None:
        """
        ایندکس را در حافظه (in-memory) با استفاده از متون، وکتورها و متادیتای داده‌شده می‌سازد.
        این متد باید self.vectorstore را مقداردهی کند.
        """
        pass

    @abstractmethod
    def save_local(self, path: str) -> None:
        """
        ایندکس ساخته‌شده در حافظه را در مسیر مشخص‌شده روی دیسک ذخیره می‌کند.
        """
        pass

    @abstractmethod
    def load_local(self, path: str) -> None:
        """
        یک ایندکس را از مسیر مشخص‌شده روی دیسک بارگذاری کرده و در self.vectorstore قرار می‌دهد.
        """
        pass