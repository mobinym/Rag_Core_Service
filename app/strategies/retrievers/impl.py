# app/strategies/retrievers/impl.py
from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores.faiss import FAISS
from langchain_ollama import OllamaLLM
import logging

from .base import BaseRetrieverStrategy
from app.models.schemas import AskResponse, SourceDocument

logger = logging.getLogger(__name__)

class BasicRetriever(BaseRetrieverStrategy):
    """یک جستجوی ساده انجام داده و پاسخ را تولید می‌کند."""
    def retrieve(self, query: str, vector_store: FAISS, llm: OllamaLLM, top_k: int) -> AskResponse:
        docs_with_scores = vector_store.similarity_search_with_score(query, k=top_k)
        retrieved_docs = [doc for doc, score in docs_with_scores]
        
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt_template = PromptTemplate.from_template("Context: {context}\n\nQuestion: {question}\n\nAnswer:")
        
        chain = prompt_template | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": query}).strip()
        
        source_documents = [
            SourceDocument(page_content=doc.page_content, metadata=doc.metadata, score=float(score))
            for doc, score in docs_with_scores
        ]
        return AskResponse(answer=answer, source_documents=source_documents)

class AdaptiveRetriever(BaseRetrieverStrategy):
    """استراتژی پیشرفته شما: بازیابی تطبیق‌پذیر."""
    def __init__(self, min_answer_length: int = 50, retry_k: int = 5):
        self.min_answer_length = min_answer_length
        self.retry_k = retry_k
        self.prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            # ✅ پرامپت دقیق و حرفه‌ای شما در اینجا استفاده می‌شود
            template="""شما یک دستیار تحلیل‌گر هستید که در پاسخ‌گویی دقیق و مستند به پرسش‌های فارسی از روی متن‌های ارائه‌شده تخصص دارید.
            هنگام پاسخ‌گویی، موارد زیر را رعایت کنید:
            - پاسخ باید دقیق، شفاف، و مبتنی بر اطلاعات صریح متن باشد
            - شامل جزئیات کلیدی مانند نام‌ها، وقایع، زمان‌ها، نقل‌قول‌ها یا نتایج مشخص باشد
            - منبع هر پاسخ را ذکر کنید (مثلاً با اشاره به عنوان بخش یا شماره صفحه، اگر موجود است)
            - هیچ‌گونه حدس، تفسیر یا اطلاعات اضافه خارج از متن نداشته باشید
            - اگر پاسخ در متن نیست، دقیقاً این جمله را بنویسید: «اطلاعاتی در متن ارائه نشده است.»
            
            پاسخ را فقط بر اساس متن زیر بنویس.
            
            سؤال: {query}
            
            متن:
            {context}
            
            پاسخ:"""
        )

    def _generate_answer(self, query: str, documents: List[Document], llm: OllamaLLM) -> str:
        if not documents:
            return "اطلاعاتی در متن ارائه نشده است."
        
        context = "\n\n".join([
            f"بخش: {doc.metadata.get('section', 'نامشخص')}, صفحه: {doc.metadata.get('page', 'نامشخص')}\n\n{doc.page_content}"
            for doc in documents
        ])
        
        chain = self.prompt_template | llm | StrOutputParser()
        return chain.invoke({"context": context, "query": query}).strip()

    def retrieve(self, query: str, vector_store: FAISS, llm: OllamaLLM, top_k: int) -> AskResponse:
        docs_with_scores = vector_store.similarity_search_with_score(query, k=top_k)
        documents = [doc for doc, score in docs_with_scores]
        answer = self._generate_answer(query, documents, llm)
        
        if len(answer) < self.min_answer_length and top_k < self.retry_k:
            logger.warning(f"پاسخ اولیه خیلی کوتاه است. تلاش مجدد با k={self.retry_k}.")
            docs_with_scores = vector_store.similarity_search_with_score(query, k=self.retry_k)
            documents = [doc for doc, score in docs_with_scores]
            answer = self._generate_answer(query, documents, llm)
        
        source_documents = [
            SourceDocument(page_content=doc.page_content, metadata=doc.metadata, score=float(score))
            for doc, score in docs_with_scores
        ]
        return AskResponse(answer=answer, source_documents=source_documents)