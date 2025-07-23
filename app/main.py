# app/main.py
import logging
import uuid
from typing import Dict, List

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from .core.config import settings
from .models.schemas import CreateSessionResponse, Chunk, AskRequest, AskResponse, SourceDocument
from .strategies.vector_stores.base import BaseVectorStoreStrategy
from .strategies.vector_stores.impl import FAISSStrategy, EMBEDDING_DIM
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
import os
# --- تنظیمات اولیه ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="RAG Core Service")

# --- دیکشنری استراتژی‌ها و حافظه ایندکس‌ها ---
STRATEGY_FACTORY: Dict[str, type[BaseVectorStoreStrategy]] = {
    "faiss": FAISSStrategy,
}
# این حافظه، ایندکس‌های ساخته شده را برای هر جلسه نگه می‌دارد
INDEX_CACHE: Dict[str, BaseVectorStoreStrategy] = {}
# --- نمونه‌سازی از LLM ---
llm = OllamaLLM(model="gemma3", base_url=settings.OLLAMA_BASE_URL)
# --- اندپوینت‌ها ---
@app.post("/sessions", response_model=CreateSessionResponse)
def create_session(
    file: UploadFile = File(...),
    extractor_strategy: str = Form("pypdf"),
    chunker_strategy: str = Form("recursive"),
    vector_store_strategy: str = Form("faiss")
):
    """
    یک فایل را دریافت، پردازش، امبد و ایندکس کرده و یک شناسه جلسه برمی‌گرداند.
    """
    # ۱. فراخوانی Document Processor Service
    logger.info("Step 1: Calling Document Processor Service...")
    files = {'file': (file.filename, file.file, file.content_type)}
    params = {'extractor_strategy': extractor_strategy, 'chunker_strategy': chunker_strategy}
    try:
        doc_response = requests.post(settings.DOCUMENT_PROCESSOR_URL, files=files, data=params)
        doc_response.raise_for_status()
        processed_data = doc_response.json()
        chunks = [Chunk(**chunk_data) for chunk_data in processed_data['chunks']]
        logger.info(f"Successfully got {len(chunks)} chunks.")
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Error communicating with Document Processor: {e}")

    if not chunks:
        raise HTTPException(status_code=400, detail="Document processing resulted in no chunks.")

    # ۲. فراخوانی Embedding Service
    logger.info("Step 2: Calling Embedding Service...")
    chunk_texts = [chunk.page_content for chunk in chunks]
    try:
        embed_response = requests.post(settings.EMBEDDING_SERVICE_URL, json={"texts": chunk_texts})
        embed_response.raise_for_status()
        vectors = embed_response.json()["vectors"]
        logger.info(f"Successfully got {len(vectors)} vectors.")
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Error communicating with Embedding Service: {e}")


    # ۳. ساخت و ذخیره ایندکس
    logger.info(f"Step 3: Creating and persisting index with '{vector_store_strategy}' strategy...")
    if vector_store_strategy not in STRATEGY_FACTORY:
        raise HTTPException(status_code=400, detail="Vector store strategy not supported.")
    
    strategy_class = STRATEGY_FACTORY[vector_store_strategy]
    index_instance = strategy_class()
    
    chunk_metadatas = [chunk.metadata for chunk in chunks]
    index_instance.create_index(texts=chunk_texts, vectors=vectors, metadatas=chunk_metadatas)

    # ۴. ذخیره ایندکس روی دیسک
    session_id = str(uuid.uuid4())
    save_path = os.path.join(settings.INDEX_DIR, session_id)
    index_instance.save_local(save_path)
    
    # (اختیاری) اضافه کردن به کش حافظه برای دسترسی فوری در درخواست‌های بعدی
    INDEX_CACHE[session_id] = index_instance
    
    return CreateSessionResponse(
        session_id=session_id,
        message="Session created and document indexed successfully.",
        total_chunks=len(chunks)
    )
#-------------------------------------------------------------------------------
# ✅ --- اندپوینت جدید برای پرسش و پاسخ ---
@app.post("/sessions/{session_id}/ask", response_model=AskResponse)
def ask_question(session_id: str, request: AskRequest):
    """Asks a question to a specific session and returns an LLM-generated answer."""
    logger.info(f"Received ask request for session '{session_id}'")
    
    # 1. Find the index: first in memory cache, then on disk
    index_instance = INDEX_CACHE.get(session_id)
    if not index_instance:
        logger.info(f"Index not in cache. Loading from disk for session: {session_id}")
        try:
            index_path = os.path.join(settings.INDEX_DIR, session_id)
            strategy_class = STRATEGY_FACTORY["faiss"]
            index_instance = strategy_class()
            index_instance.load_local(index_path)
            INDEX_CACHE[session_id] = index_instance
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Session ID not found on disk.")
    
    # 2. Get the query vector from the embedding service
    logger.info(f"Getting embedding for query: '{request.query}'")
    try:
        embed_response = requests.post(settings.EMBEDDING_SERVICE_URL, json={"texts": [request.query]})
        embed_response.raise_for_status()
        query_vector = embed_response.json()["vectors"][0]
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Error communicating with Embedding Service: {e}")

    # 3. Retrieve relevant documents from the vector store
    logger.info("Retrieving relevant documents from vector store...")
    retrieved_docs = index_instance.search(query_vector, k=request.top_k)
    if not retrieved_docs:
        raise HTTPException(status_code=404, detail="No relevant documents found for the query.")

    # 4. Create the context for the LLM
    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # 5. Create the final prompt for the LLM
    prompt_template = f"""
    Based on the following context, please provide a concise and helpful answer in Persian to the user's question.
    If the context does not contain the answer, state that the answer is not in the provided documents.

    Context:
    ---
    {context}
    ---

    Question: {request.query}
    Answer (in Persian):
    """
    
    # 6. Call the LLM to generate the answer
    logger.info("Generating final answer with LLM...")
    try:
        answer = llm.invoke(prompt_template).strip()
    except Exception as e:
        logger.error(f"Error invoking LLM: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate answer from LLM.")
    
    # 7. Prepare the final response
    source_documents = [
        SourceDocument(
            page_content=doc.page_content,
            metadata=doc.metadata,
            # ✅ Here is the fix: explicitly cast the score to a standard Python float
            score=float(doc.metadata.get('score', 0.0))
        )
        for doc in retrieved_docs
    ]
    
    return AskResponse(answer=answer, source_documents=source_documents)