# app/strategies/retrievers/impl.py

import requests
from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores.faiss import FAISS
from langchain_ollama import OllamaLLM
import logging
from sentence_transformers import SentenceTransformer # ✅ ایمپورت جدید

from .base import BaseRetrieverStrategy
from app.models.schemas import AskResponse, SourceDocument
from app.core.config import settings

logger = logging.getLogger(__name__)

print("LOADING LOCAL BGE-M3 MODEL FOR DEBUGGING...")
local_embedding_model = SentenceTransformer('BAAI/bge-m3')
print("LOCAL MODEL LOADED.")
def get_embedding_for_query_LOCAL_TEST(query: str) -> List[float]:
    """
    (تست موقت) کوئری را به صورت لوکال و با نرمال‌سازی صحیح امبد می‌کند.
    """
    print("DEBUG: Using LOCAL embedding function for query.")
    return local_embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()


def get_embedding_for_query(query: str) -> List[float]:
    """
    سرویس امبدینگ خارجی را برای دریافت وکتور یک کوئری فراخوانی می‌کند.
    """
    try:
        url = settings.services.embedding_service_url
        payload = {"texts": [query]}
        
        # ✅ اضافه کردن لاگ برای دیباگ
        print(f"DEBUG: Sending query '{query}' to embedding service at '{url}'")
        print(f"DEBUG: Payload: {payload}")
        
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        response_json = response.json()
        print(f"DEBUG: Received response from embedding service: {response_json}")
        
        vector = response_json["vectors"][0]
        print(f"DEBUG: Extracted vector starts with: {str(vector)[:80]}...") # نمایش ۸۰ کاراکتر اول وکتور
        
        return vector

    except requests.RequestException as e:
        logger.error(f"Failed to get embedding for query: {e}")
        print(f"DEBUG ERROR: Could not connect to embedding service: {e}") # ✅ لاگ خطا
        raise RuntimeError(f"Could not connect to embedding service: {e}")
    except (KeyError, IndexError) as e:
        logger.error(f"Invalid response from embedding service: {e}")
        print(f"DEBUG ERROR: Invalid response from embedding service: {e}") # ✅ لاگ خطا
        raise RuntimeError(f"Invalid response from embedding service: {e}")


class BasicRetriever(BaseRetrieverStrategy):
    """اسناد را با دریافت امبدینگ کوئری از سرویس خارجی بازیابی می‌کند."""
    def retrieve(self, query: str, vector_store: FAISS, llm: OllamaLLM, top_k: int) -> AskResponse:
        # ۱. دریافت وکتور کوئری از سرویس امبدینگ شما
        query_vector = get_embedding_for_query_LOCAL_TEST(query)

        # ۲. انجام جستجو با استفاده از وکتور به جای متن
        docs_with_scores = vector_store.similarity_search_with_score_by_vector(
            embedding=query_vector,
            k=top_k
        )
        
        # ۳. تولید پاسخ (بدون تغییر)
        context = "\n\n---\n\n".join([doc.page_content for doc, score in docs_with_scores])
        prompt_template = PromptTemplate.from_template("Context: {context}\n\nQuestion: {question}\n\nAnswer:")
        chain = prompt_template | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": query}).strip()
        
        source_documents = [SourceDocument(page_content=doc.page_content, metadata=doc.metadata, score=float(score)) for doc, score in docs_with_scores]
        return AskResponse(answer=answer, source_documents=source_documents)


class AdaptiveRetriever(BaseRetrieverStrategy):
    """Retriever پیشرفته، که اکنون از سرویس امبدینگ خارجی برای کوئری‌ها استفاده می‌کند."""
    def __init__(self):
        self.min_answer_length = settings.retriever_settings.adaptive.min_answer_length
        self.retry_k = settings.retriever_settings.adaptive.retry_k
        self.prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template="""[INST]
You are an expert Persian-speaking analytical assistant. Your task is to accurately answer questions based *only* on the provided context.

Follow these steps to generate your answer:
1.  **Analyze the User's Question:** Understand the core intent of the user's query: '{query}'.
2.  **Scan the Context:** Carefully read the entire provided context to find the most relevant sentences or data points that directly answer the question.
3.  **Synthesize the Answer:** Formulate a clear, concise, and direct answer in Persian based *exclusively* on the information you found.
4.  **Cite Sources (if available):** If the context provides clear source information like page or section numbers for the relevant data, mention it briefly in parentheses at the end of your answer.
5.  **Fallback:** If after careful analysis you cannot find the answer within the context, respond with *only* this exact phrase: "اطلاعاتی در متن ارائه نشده است."

**Context:**
---
{context}
---

**User's Question:** {query}
[/INST]

**Answer:**
"""
        )


    def _generate_answer(self, query: str, documents: List[Document], llm: OllamaLLM) -> str:
        if not documents:
            return "اطلاعاتی در متن ارائه نشده است."
        context = "\n\n---\n\n".join([doc.page_content for doc in documents])
        chain = self.prompt_template | llm | StrOutputParser()
        return chain.invoke({"context": context, "query": query}).strip()

    def retrieve(self, query: str, vector_store: FAISS, llm: OllamaLLM, top_k: int) -> AskResponse:
        # ۱. دریافت وکتور کوئری از سرویس امبدینگ
        query_vector = get_embedding_for_query_LOCAL_TEST(query)
        
        # ۲. انجام جستجو با استفاده از وکتور
        docs_with_scores = vector_store.similarity_search_with_score_by_vector(
            embedding=query_vector,
            k=top_k
        )
            # ✅ کد دیباگ: چاپ محتوای اسنادی که بازیابی شده‌اند
        print("\n--- DEBUG: RETRIEVED DOCUMENTS ---")
        if not docs_with_scores:
            print("!!! NO DOCUMENTS RETRIEVED !!!")
        else:
            for i, (doc, score) in enumerate(docs_with_scores):
                print(f"--- Document {i+1} | Score: {score} ---")
                print(doc.page_content)
                print("-" * 30)
        print("--- END DEBUG ---\n")

        documents = [doc for doc, score in docs_with_scores]
        answer = self._generate_answer(query, documents, llm)
        
        if len(answer) < self.min_answer_length and top_k < self.retry_k:
            logger.warning(f"پاسخ اولیه خیلی کوتاه است. تلاش مجدد با k={self.retry_k}.")
            docs_with_scores = vector_store.similarity_search_with_score_by_vector(
                embedding=query_vector,
                k=self.retry_k
            )
            documents = [doc for doc, score in docs_with_scores]
            answer = self._generate_answer(query, documents, llm)
            
        source_documents = [SourceDocument(page_content=doc.page_content, metadata=doc.metadata, score=float(score)) for doc, score in docs_with_scores]
        return AskResponse(answer=answer, source_documents=source_documents)