# app/strategies/retrievers/impl.py

import requests
from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS, Chroma
from langchain_ollama import OllamaLLM
import logging

from .base import BaseRetrieverStrategy
from app.models.schemas import AskResponse, SourceDocument
from app.core.config import settings

logger = logging.getLogger(__name__)


def get_embedding_for_query(query: str) -> List[float]:

    try:
        response = requests.post(settings.services.embedding_service_url, json={"texts": [query]}, timeout=60)
        response.raise_for_status()
        return response.json()["vectors"][0]
    except requests.RequestException as e:
        logger.error(f"Failed to get embedding for query: {e}")
        raise RuntimeError(f"Could not connect to embedding service: {e}")
    except (KeyError, IndexError) as e:
        logger.error(f"Invalid response from embedding service: {e}")
        raise RuntimeError(f"Invalid response from embedding service: {e}")


class BasicRetriever(BaseRetrieverStrategy):
    def retrieve(self, query: str, vector_store: FAISS, llm: OllamaLLM, top_k: int) -> AskResponse:
        query_vector = get_embedding_for_query(query)

        docs_with_scores = vector_store.similarity_search_with_score_by_vector(
            embedding=query_vector,
            k=top_k
        )
        
        context = "\n\n---\n\n".join([doc.page_content for doc, score in docs_with_scores])
        prompt_template = PromptTemplate.from_template("Context: {context}\n\nQuestion: {question}\n\nAnswer:")
        chain = prompt_template | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": query}).strip()
        
        source_documents = [SourceDocument(page_content=doc.page_content, metadata=doc.metadata, score=float(score)) for doc, score in docs_with_scores]
        return AskResponse(answer=answer, source_documents=source_documents)


class AdaptiveRetriever(BaseRetrieverStrategy):
    """Retriever پیشرفته، که از سرویس امبدینگ خارجی برای کوئری‌ها استفاده می‌کند."""
    def __init__(self):
        self.min_answer_length = settings.retriever_settings.adaptive.min_answer_length
        self.retry_k = settings.retriever_settings.adaptive.retry_k
        self.prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template="""[INST]
شما یک دستیار تحلیلی فارسی‌زبان و متخصص هستید. وظیفه شما این است که فقط بر اساس **متن ارائه‌شده**، به سؤالات به‌دقت و با دقت پاسخ دهید.

برای تولید پاسخ، مراحل زیر را دنبال کنید:

1.  **تحلیل پرسش کاربر:** هدف اصلی پرسش زیر را به‌درستی درک کنید: «{query}».
2.  **بررسی دقیق متن:** تمام متن داده‌شده را با دقت بخوانید و جملات یا داده‌هایی را که مستقیماً به پرسش پاسخ می‌دهند، پیدا کنید.
3.  **تدوین پاسخ:** یک پاسخ روشن، مختصر و مستقیم به زبان فارسی بنویسید که فقط بر اساس اطلاعات موجود در متن باشد.
4.  **ارجاع به منبع (در صورت وجود):** اگر در متن اطلاعات منبع مثل شماره صفحه یا بخش آمده باشد، در پایان پاسخ به‌صورت کوتاه به آن اشاره کنید (مثلاً: «(صفحه ۱۲)»).
5.  **در صورت نبود اطلاعات:** اگر پس از بررسی دقیق متن، پاسخی برای پرسش در آن پیدا نکردید، دقیقاً فقط این عبارت را بنویسید: «اطلاعاتی در متن ارائه نشده است.»

**متن:**
---
{context}
---

**پرسش کاربر:** {query}
[/INST]

**پاسخ:**
"""
        )

    def _generate_answer(self, query: str, documents: List[Document], llm: OllamaLLM) -> str:
        if not documents:
            return "اطلاعاتی در متن ارائه نشده است."
        context = "\n\n---\n\n".join([doc.page_content for doc in documents])
        chain = self.prompt_template | llm | StrOutputParser()
        return chain.invoke({"context": context, "query": query}).strip()

    def retrieve(self, query: str, vector_store: FAISS, llm: OllamaLLM, top_k: int) -> AskResponse:
        query_vector = get_embedding_for_query(query)
        
        docs_with_scores = []
        if isinstance(vector_store, FAISS):
            logger.info("Using FAISS search method.")
            docs_with_scores = vector_store.similarity_search_with_score_by_vector(
                embedding=query_vector,
                k=top_k
            )
        elif isinstance(vector_store, Chroma):
            logger.info("Using Chroma search method.")
            retrieved_docs = vector_store.similarity_search_by_vector(
                embedding=query_vector,
                k=top_k
            )
            docs_with_scores = [(doc, 0.0) for doc in retrieved_docs]
        else:
            raise TypeError(f"Unsupported vector store type: {type(vector_store)}")

        documents = [doc for doc, score in docs_with_scores]
        answer = self._generate_answer(query, documents, llm)
        
        if len(answer) < self.min_answer_length and top_k < self.retry_k:
            logger.warning(f"پاسخ اولیه خیلی کوتاه است. تلاش مجدد با k={self.retry_k}.")
            if isinstance(vector_store, FAISS):
                docs_with_scores = vector_store.similarity_search_with_score_by_vector(embedding=query_vector, k=self.retry_k)
            elif isinstance(vector_store, Chroma):
                retrieved_docs = vector_store.similarity_search_by_vector(embedding=query_vector, k=self.retry_k)
                docs_with_scores = [(doc, 0.0) for doc in retrieved_docs]

            documents = [doc for doc, score in docs_with_scores]
            answer = self._generate_answer(query, documents, llm)
            
        source_documents = [SourceDocument(page_content=doc.page_content, metadata=doc.metadata, score=float(score)) for doc, score in docs_with_scores]
        return AskResponse(answer=answer, source_documents=source_documents)