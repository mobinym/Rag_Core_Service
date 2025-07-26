# rag_core_service/app/main.py

import logging
import uuid
import os
import json
from typing import Dict, Set

import requests
from fastapi import FastAPI, UploadFile, File, Form, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from .errors import ServiceException, ERROR_CODES
from .core.config import settings
# ✅ ایمپورت کردن مدل‌های پاسخ جدید و قدیمی
from .models.schemas import CreateSessionResponse, Chunk, AskRequest, AskResponse, FormattedAskResponse
from .strategies.vector_stores.base import BaseVectorStoreStrategy
from .strategies.vector_stores.impl import FAISSStrategy, ChromaStrategy
from .strategies.retrievers.base import BaseRetrieverStrategy
from .strategies.retrievers.impl import BasicRetriever, AdaptiveRetriever
from langchain_ollama import OllamaLLM

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Core Service",
    description="A flexible RAG service with multiple strategies and structured error handling.",
    version="1.0"
)


# --- Exception Handlers (بدون تغییر) ---
@app.exception_handler(ServiceException)
async def service_exception_handler(request: Request, exc: ServiceException):
    return JSONResponse(status_code=exc.status_code, content={"success": False, "error": {"code": exc.error_code, "message": exc.message, "details": exc.details}})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    clean_errors = [f"Field '{' -> '.join(map(str, e['loc'][1:]))}': {e['msg']}" for e in exc.errors()]
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"success": False, "error": {"code": 90001, "message": ERROR_CODES[90001], "details": clean_errors}})

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"An unexpected error occurred: {exc}", exc_info=True)
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"success": False, "error": {"code": 99999, "message": ERROR_CODES[99999]}})


# --- ✅ Helper Function for Formatting ---
def format_rag_response(raw_response: AskResponse) -> str:
    """
    پاسخ خام RAG را به یک متن فرمت‌شده و تمیز Markdown تبدیل می‌کند.
    """
    answer = raw_response.answer
    source_pages: Set[int] = set()
    for doc in raw_response.source_documents:
        if "page" in doc.metadata and doc.metadata["page"] is not None:
            source_pages.add(doc.metadata["page"])
            
    formatted_output = f"**پاسخ:**\n\n{answer}"
    
    if source_pages:
        formatted_output += "\n\n---\n\n**منابع:**\n"
        for page_num in sorted(list(source_pages)):
            formatted_output += f"\n* صفحه {page_num}"
            
    return formatted_output


# --- Service Configuration ---
RETRIEVERS: Dict[str, BaseRetrieverStrategy] = {"basic": BasicRetriever(), "adaptive": AdaptiveRetriever()}
VECTOR_STORE_FACTORY: Dict[str, type[BaseVectorStoreStrategy]] = {"faiss": FAISSStrategy, "chroma": ChromaStrategy}
INDEX_CACHE: Dict[str, BaseVectorStoreStrategy] = {}
llm = OllamaLLM(model="qwen3:4b", base_url=settings.OLLAMA_BASE_URL)


# --- API Endpoints ---
@app.post("/sessions", response_model=CreateSessionResponse)
def create_session(
    file: UploadFile = File(..., description="The document file to be processed (PDF, DOCX, etc.)"),
    extractor_strategy: str = Form("pdfFA",enum=["docling", "pypdf", "pdfFA", "docx"], description="Text extraction strategy(Supported: docling, pypdf, pdfFA, docx)"),
    chunker_strategy: str = Form("token_based",enum=["recursive", "custom_sentence", "ollama_semantic", "ollama_semantic_p", "token_based"],description="Text chunking strategy(Supported: recursive, custom_sentence, ollama_semantic, ollama_semantic_p, token_based)"),
    vector_store_strategy: str = Form("faiss", enum=["faiss", "chroma"], description="The vector store strategy to use")
):
    # (این اندپوینت بدون تغییر باقی می‌ماند)
    logger.info("Step 1: Calling Document Processor Service...")
    try:
        files = {'file': (file.filename, file.file, file.content_type)}
        params = {'extractor_strategy': extractor_strategy, 'chunker_strategy': chunker_strategy}
        doc_response = requests.post(settings.DOCUMENT_PROCESSOR_URL, files=files, data=params, timeout=300)
        doc_response.raise_for_status()
        processed_data = doc_response.json()
        chunks = [Chunk(**chunk_data) for chunk_data in processed_data['chunks']]
        logger.info(f"Successfully received {len(chunks)} chunks.")
    except requests.RequestException as e:
        raise ServiceException(status_code=503, error_code=40001, message=f"The Document Processor service is unavailable: {e}")
    if not chunks:
        raise ServiceException(status_code=400, error_code=30003, message=ERROR_CODES[30003])
    logger.info("Step 2: Calling Embedding Service...")
    try:
        chunk_texts = [chunk.chunk_content for chunk in chunks]
        embed_response = requests.post(settings.EMBEDDING_SERVICE_URL, json={"texts": chunk_texts}, timeout=180)
        embed_response.raise_for_status()
        vectors = embed_response.json()["vectors"]
        logger.info(f"Successfully received {len(vectors)} vectors.")
    except requests.RequestException as e:
        raise ServiceException(status_code=503, error_code=40001, message=f"The Embedding service is unavailable: {e}")
    logger.info(f"Step 3: Creating index with '{vector_store_strategy}' strategy...")
    strategy_class = VECTOR_STORE_FACTORY.get(vector_store_strategy)
    if not strategy_class:
        raise ServiceException(status_code=400, error_code=30001, message=f"Vector store strategy '{vector_store_strategy}' is not supported.")
    try:
        index_instance = strategy_class()
        chunk_metadatas = [chunk.metadata for chunk in chunks]
        index_instance.create_index(texts=chunk_texts, vectors=vectors, metadatas=chunk_metadatas)
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(settings.INDEX_DIR, session_id)
        index_path = os.path.join(session_dir, "index")
        index_instance.save_local(index_path)
        config_path = os.path.join(session_dir, "index_config.json")
        with open(config_path, 'w') as f:
            json.dump({"vector_store_strategy": vector_store_strategy}, f)
        INDEX_CACHE[session_id] = index_instance
    except Exception as e:
        logger.error(f"Failed during index creation or saving: {e}", exc_info=True)
        raise ServiceException(status_code=500, error_code=40002, message=f"Failed to create or save the index: {e}")
    return CreateSessionResponse(session_id=session_id, message="Session created and document indexed successfully.", total_chunks=len(chunks))


@app.post("/sessions/{session_id}/ask", response_model=FormattedAskResponse) # ✅ تغییر مدل پاسخ
def ask_question(session_id: str, request: AskRequest):
    """Answers a question and returns a clean, formatted response."""
    logger.info(f"Received ask request for session '{session_id}' with strategy '{request.retrieval_strategy}'")
    
    # (بخش بارگذاری ایندکس بدون تغییر باقی می‌ماند)
    index_instance = INDEX_CACHE.get(session_id)
    if not index_instance:
        logger.info(f"Index not in cache. Loading from disk for session: {session_id}")
        try:
            session_dir = os.path.join(settings.INDEX_DIR, session_id)
            config_path = os.path.join(session_dir, "index_config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
            vector_store_strategy = config.get("vector_store_strategy")
            if not vector_store_strategy:
                 raise ServiceException(status_code=404, error_code=30004, message="Session is corrupted: vector store strategy is unknown.")
            strategy_class = VECTOR_STORE_FACTORY[vector_store_strategy]
            index_instance = strategy_class()
            index_path = os.path.join(session_dir, "index")
            index_instance.load_local(index_path)
            INDEX_CACHE[session_id] = index_instance
            logger.info(f"Successfully loaded index with '{vector_store_strategy}' strategy.")
        except FileNotFoundError:
            raise ServiceException(status_code=404, error_code=30002, message=f"Session ID '{session_id}' not found on disk.")
        except Exception as e:
            raise ServiceException(status_code=500, error_code=40003, message=f"Failed to load the index from disk: {e}")
    if not index_instance.vectorstore:
        raise ServiceException(status_code=500, error_code=30004, message="Vector store for this session is available but not initialized correctly.")

    retriever = RETRIEVERS.get(request.retrieval_strategy)
    if not retriever:
        raise ServiceException(status_code=400, error_code=30001, message=f"Retrieval strategy '{request.retrieval_strategy}' is not supported.")
    
    try:
        # ✅ مرحله ۱: دریافت پاسخ خام از retriever
        raw_response: AskResponse = retriever.retrieve(query=request.query, vector_store=index_instance.vectorstore, llm=llm, top_k=request.top_k)
        
        # ✅ مرحله ۲: فرمت‌بندی پاسخ خام
        formatted_string = format_rag_response(raw_response)
        
        # ✅ مرحله ۳: بازگرداندن خروجی تمیز در مدل جدید
        return FormattedAskResponse(formatted_answer=formatted_string)

    except Exception as e:
        logger.error(f"Error during retrieval with '{request.retrieval_strategy}': {e}", exc_info=True)
        raise ServiceException(status_code=500, error_code=40004, message=f"An error occurred during the retrieval and generation process: {e}")