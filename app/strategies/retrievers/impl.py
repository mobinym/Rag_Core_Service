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
    
class HybridRRFRetriever:

    def retrieve_documents(self, 
                           query: str, 
                           index: Union[FAISSStrategy, ChromaStrategy], 
                           top_k: int,
                           strategy_name: str,  
                           vector_search_type: str = "similarity",
                           weights: List[float] = [0.5, 0.5]
                           ) -> RetrieveResponse:

        if not isinstance(index, FAISSStrategy):
            logger.warning("HybridRRFRetriever only supports modified FAISSStrategy. Skipping.")
            raise NotImplementedError("HybridRRFRetriever only supports FAISSStrategy")

        if not index.vectorstore:
            raise RuntimeError("FAISS vectorstore is not loaded in the index strategy.")
            
        
        search_kwargs = {"k": top_k}
        if vector_search_type == "mmr":
            search_kwargs["fetch_k"] = top_k * 5 
            logger.info(f"Using vector search: MMR (k={top_k}, fetch_k={search_kwargs['fetch_k']})")
        else:
            logger.info(f"Using vector search: Similarity (k={top_k})")

        vector_retriever = index.vectorstore.as_retriever(
            search_type=vector_search_type,
            search_kwargs=search_kwargs
        )
        # -------------------------
            
        if not index.bm25_retriever:
            logger.warning(f"BM25Retriever not found. Falling back to vector search only (type: {vector_search_type}).")
            docs = vector_retriever.invoke(query) 

        else:
            logger.info(f"Performing Hybrid RRF Search (BM25 weight={weights[0]}, Vector weight={weights[1]})")
            
            index.bm25_retriever.k = top_k 
            
            ensemble_retriever = EnsembleRetriever(
                retrievers=[index.bm25_retriever, vector_retriever], 
                weights=weights 
            )
            
            hybrid_docs = ensemble_retriever.invoke(query) 
            
            docs = hybrid_docs[:top_k]

        source_documents = []
        for doc in docs:
            source_documents.append(
                SourceDocument(
                    page_content=doc.page_content,
                    metadata=doc.metadata,
                    score=0.0  
                )
            )
            
        return RetrieveResponse(
            source_documents=source_documents
        )