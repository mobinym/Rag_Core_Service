# app/strategies/retrievers/base.py
from abc import ABC, abstractmethod
from langchain_community.vectorstores.faiss import FAISS
from langchain_ollama import OllamaLLM
from app.models.schemas import AskResponse

class BaseRetrieverStrategy(ABC):
    @abstractmethod
    def retrieve(self, query: str, vector_store: FAISS, llm: OllamaLLM, top_k: int) -> AskResponse:
        pass