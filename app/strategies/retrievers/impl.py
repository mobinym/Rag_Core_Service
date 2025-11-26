# app/strategies/retrievers/impl.py

import requests
from typing import List, Union
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
import logging
from langchain_core.vectorstores import VectorStore 
from langchain_chroma import Chroma
from .base import BaseRetrieverStrategy
from langchain.retrievers import EnsembleRetriever
from app.models.schemas import AskResponse, SourceDocument, RetrieveResponse 
from app.core.config import settings
from ..vector_stores.impl import FAISSStrategy, ChromaStrategy
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
    def retrieve(self, query: str, vector_store: VectorStore, llm: OllamaLLM, top_k: int) -> AskResponse:
        query_vector = get_embedding_for_query(query)

        docs_with_scores = []
        if isinstance(vector_store, FAISS):
            docs_with_scores = vector_store.similarity_search_with_score_by_vector(embedding=query_vector, k=top_k)
        elif isinstance(vector_store, Chroma):
            retrieved_docs = vector_store.similarity_search_by_vector(embedding=query_vector, k=top_k)
            docs_with_scores = [(doc, 0.0) for doc in retrieved_docs]
        else:
            raise TypeError(f"Unsupported vector store type: {type(vector_store)}")

        
        context = "\n\n---\n\n".join([doc.page_content for doc, score in docs_with_scores])
        prompt_template = PromptTemplate.from_template("Context: {context}\n\nQuestion: {question}\n\nAnswer:")
        chain = prompt_template | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": query}).strip()
        
        source_documents = [SourceDocument(page_content=doc.page_content, metadata=doc.metadata, score=float(score)) for doc, score in docs_with_scores]
        return AskResponse(answer=answer, source_documents=source_documents)
    
    def retrieve_documents(self, query: str, vector_store: VectorStore, top_k: int) -> RetrieveResponse:
        """فقط داکیومنت‌های مرتبط را بر اساس کوئری بازیابی می‌کند."""
        query_vector = get_embedding_for_query(query)

        docs_with_scores = []
        if isinstance(vector_store, FAISS):
            docs_with_scores = vector_store.similarity_search_with_score_by_vector(embedding=query_vector, k=top_k)
        elif isinstance(vector_store, Chroma):
            retrieved_docs = vector_store.similarity_search_by_vector(embedding=query_vector, k=top_k)
            docs_with_scores = [(doc, 0.0) for doc in retrieved_docs]
        else:
            raise TypeError(f"Unsupported vector store type: {type(vector_store)}")

        source_documents = [
            SourceDocument(page_content=doc.page_content, metadata=doc.metadata, score=float(score))
            for doc, score in docs_with_scores
        ]
        
        return RetrieveResponse(source_documents=source_documents)

class AdaptiveRetriever(BaseRetrieverStrategy):
    """Retriever پیشرفته، که از سرویس امبدینگ خارجی برای کوئری‌ها استفاده می‌کند."""
    def __init__(self):
        self.min_answer_length = settings.retriever_settings.adaptive.min_answer_length
        self.retry_k = settings.retriever_settings.adaptive.retry_k
        self.prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template=settings.prompt_templates.adaptive_retriever)

    def _generate_answer(self, query: str, documents: List[Document], llm: OllamaLLM) -> str:
        if not documents:
            return "اطلاعاتی در متن ارائه نشده است."
        context = "\n\n---\n\n".join([doc.page_content for doc in documents])
        chain = self.prompt_template | llm | StrOutputParser()
        return chain.invoke({"context": context, "query": query}).strip()

    def retrieve(self, query: str, vector_store: VectorStore, llm: OllamaLLM, top_k: int) -> AskResponse:
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
    
    def retrieve_documents(self, query: str, vector_store: VectorStore, top_k: int) -> RetrieveResponse:
        """فقط داکیومنت‌های مرتبط را بر اساس کوئری بازیابی می‌کند."""
        query_vector = get_embedding_for_query(query)

        docs_with_scores = []
        if isinstance(vector_store, FAISS):
            docs_with_scores = vector_store.similarity_search_with_score_by_vector(embedding=query_vector, k=top_k)
        elif isinstance(vector_store, Chroma):
            retrieved_docs = vector_store.similarity_search_by_vector(embedding=query_vector, k=top_k)
            docs_with_scores = [(doc, 0.0) for doc in retrieved_docs]
        else:
            raise TypeError(f"Unsupported vector store type: {type(vector_store)}")

        source_documents = [
            SourceDocument(page_content=doc.page_content, metadata=doc.metadata, score=float(score))
            for doc, score in docs_with_scores
        ]
        
        return RetrieveResponse(source_documents=source_documents)
    
class HybridRRFRetriever(BaseRetrieverStrategy):
    def __init__(self, weights: List[float] = None, vector_search_type: str = "mmr"):
        self.weights = weights or [0.2, 0.8] 
        self.vector_search_type = vector_search_type

    def retrieve(self, query: str, vector_store: VectorStore, llm: OllamaLLM, top_k: int) -> AskResponse:
        retrieve_response = self.retrieve_documents(query, vector_store, top_k)
        docs = retrieve_response.source_documents

        if not docs:
            return AskResponse(answer="اطلاعاتی یافت نشد.", source_documents=[])


        context_docs = [doc for doc in docs] 
        context = "\n\n---\n\n".join([doc.page_content for doc in context_docs])
        
        prompt = PromptTemplate.from_template("Context: {context}\n\nQuestion: {question}\n\nAnswer:")
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": query}).strip()

        return AskResponse(answer=answer, source_documents=docs)

    def retrieve_documents(self, 
                           query: str, 
                           vector_store: Union[VectorStore, object], 
                           top_k: int, 
                           strategy_name: str = "hybrid_rrf",
                           vector_search_type: str = "mmr",
                           weights: List[float] = None
                           ) -> RetrieveResponse:
        
        current_weights = weights or self.weights
        

        bm25_retriever = None
        target_vector_store = vector_store

        if hasattr(vector_store, "bm25_retriever"):
            bm25_retriever = vector_store.bm25_retriever
            if hasattr(vector_store, "vectorstore"):
                target_vector_store = vector_store.vectorstore
        
        if not isinstance(target_vector_store, FAISS):
            logger.warning("Hybrid search fallback: Vector store is not FAISS. Falling back to vector search only.")
            if hasattr(target_vector_store, "as_retriever"):
                 retriever = target_vector_store.as_retriever(search_kwargs={"k": top_k})
                 docs = retriever.invoke(query)
                 return self._format_docs(docs)
            else:
                 raise ValueError("Invalid vector store provided for Hybrid search.")

        search_kwargs = {"k": top_k}
        if vector_search_type == "mmr":
            search_kwargs["fetch_k"] = top_k * 5
        
        vector_retriever = target_vector_store.as_retriever(
            search_type=vector_search_type,
            search_kwargs=search_kwargs
        )

        if not bm25_retriever:
            logger.warning("BM25Retriever not found on the provided store. Performing Standard Vector Search instead.")
            docs = vector_retriever.invoke(query)
        else:
            logger.info(f"Performing Hybrid RRF Search (BM25={current_weights[0]}, Vector={current_weights[1]})")
            bm25_retriever.k = top_k
            
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=current_weights
            )
            
            docs = ensemble_retriever.invoke(query)
            docs = docs[:top_k]

        return self._format_docs(docs)

    def _format_docs(self, docs: List[Document]) -> RetrieveResponse:
        """helper برای فرمت‌دهی خروجی"""
        source_documents = []
        for doc in docs:
            source_documents.append(
                SourceDocument(
                    page_content=doc.page_content,
                    metadata=doc.metadata,
                    score=0.0 
                )
            )
        return RetrieveResponse(source_documents=source_documents)