# app/main.py
import logging
import uuid
import os
from typing import Dict

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from .core.config import settings
from .models.schemas import CreateSessionResponse, Chunk, AskRequest, AskResponse
from .strategies.vector_stores.base import BaseVectorStoreStrategy
from .strategies.vector_stores.impl import FAISSStrategy
from .strategies.retrievers.base import BaseRetrieverStrategy
from .strategies.retrievers.impl import BasicRetriever, AdaptiveRetriever
from langchain_ollama import OllamaLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="RAG Core Service")

RETRIEVERS: Dict[str, BaseRetrieverStrategy] = {
    "basic": BasicRetriever(),
    "adaptive": AdaptiveRetriever()
}
VECTOR_STORE_FACTORY: Dict[str, type[BaseVectorStoreStrategy]] = {"faiss": FAISSStrategy}
INDEX_CACHE: Dict[str, BaseVectorStoreStrategy] = {}
llm = OllamaLLM(model="gemma3", base_url=settings.OLLAMA_BASE_URL)

@app.post("/sessions", response_model=CreateSessionResponse)
def create_session(
    file: UploadFile = File(...),
    extractor_strategy: str = Form("pypdf"),
    chunker_strategy: str = Form("recursive"),
    vector_store_strategy: str = Form("faiss")
):
    logger.info("Step 1: Calling Document Processor Service...")
    files = {'file': (file.filename, file.file, file.content_type)}
    params = {'extractor_strategy': extractor_strategy, 'chunker_strategy': chunker_strategy}
    try:
        doc_response = requests.post(settings.DOCUMENT_PROCESSOR_URL, files=files, data=params, timeout=300)
        doc_response.raise_for_status()
        processed_data = doc_response.json()
        chunks = [Chunk(**chunk_data) for chunk_data in processed_data['chunks']]
        logger.info(f"Successfully got {len(chunks)} chunks.")
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Error communicating with Document Processor: {e}")

    if not chunks:
        raise HTTPException(status_code=400, detail="Document processing resulted in no chunks.")

    logger.info("Step 2: Calling Embedding Service...")
    # ✅ اصلاح نام فیلد برای استخراج صحیح متن چانک‌ها
    chunk_texts = [chunk.chunk_content for chunk in chunks]
    try:
        embed_response = requests.post(settings.EMBEDDING_SERVICE_URL, json={"texts": chunk_texts}, timeout=120)
        embed_response.raise_for_status()
        vectors = embed_response.json()["vectors"]
        logger.info(f"Successfully got {len(vectors)} vectors.")
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Error communicating with Embedding Service: {e}")

    logger.info(f"Step 3: Creating index with '{vector_store_strategy}' strategy...")
    strategy_class = VECTOR_STORE_FACTORY.get(vector_store_strategy)
    if not strategy_class:
        raise HTTPException(status_code=400, detail="Vector store strategy not supported.")
    
    index_instance = strategy_class()
    chunk_metadatas = [chunk.metadata for chunk in chunks]
    index_instance.create_index(texts=chunk_texts, vectors=vectors, metadatas=chunk_metadatas)

    session_id = str(uuid.uuid4())
    save_path = os.path.join(settings.INDEX_DIR, session_id)
    index_instance.save_local(save_path)
    INDEX_CACHE[session_id] = index_instance
    
    return CreateSessionResponse(
        session_id=session_id,
        message="Session created and document indexed successfully.",
        total_chunks=len(chunks)
    )


@app.post("/sessions/{session_id}/ask", response_model=AskResponse)
def ask_question(session_id: str, request: AskRequest):
    logger.info(f"Received ask request for session '{session_id}' with strategy '{request.retrieval_strategy}'")
    
    index_instance = INDEX_CACHE.get(session_id)
    if not index_instance:
        logger.info(f"Index not in cache. Loading from disk for session: {session_id}")
        try:
            index_path = os.path.join(settings.INDEX_DIR, session_id)
            strategy_class = VECTOR_STORE_FACTORY["faiss"]
            index_instance = strategy_class()
            index_instance.load_local(index_path)
            INDEX_CACHE[session_id] = index_instance
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Session ID not found on disk.")
    
    vector_store = index_instance.vectorstore
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store for this session is not available.")

    retriever = RETRIEVERS.get(request.retrieval_strategy)
    if not retriever:
        raise HTTPException(status_code=400, detail=f"Retrieval strategy '{request.retrieval_strategy}' not supported.")
    
    try:
        return retriever.retrieve(query=request.query, vector_store=vector_store, llm=llm, top_k=request.top_k)
    except Exception as e:
        logger.error(f"Error during retrieval with '{request.retrieval_strategy}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during the retrieval and generation process.")