# app/strategies/retrievers/base.py
from abc import ABC, abstractmethod
# ✅ ایمپورت کردن کلاس پایه VectorStore
from langchain_core.vectorstores import VectorStore
from langchain_ollama import OllamaLLM
from app.models.schemas import AskResponse

class BaseRetrieverStrategy(ABC):
    @abstractmethod
    # ✅ تغییر FAISS به VectorStore
    def retrieve(self, query: str, vector_store: VectorStore, llm: OllamaLLM, top_k: int) -> AskResponse:
        pass