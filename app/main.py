# rag_core_service/app/main.py

import logging
import uuid
import os
import json
from typing import Dict, Set
from .monitoring import monitoring_logger
import requests
from fastapi import FastAPI, UploadFile, File, Form, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import time
from .errors import ServiceException, ERROR_CODES
from .core.config import settings
from .models.schemas import CreateSessionResponse, AskRequest, AskResponse, StructuredAskResponse, Chunk 
from .strategies.vector_stores.base import BaseVectorStoreStrategy
from .strategies.vector_stores.impl import FAISSStrategy, ChromaStrategy
from .strategies.retrievers.base import BaseRetrieverStrategy
from .strategies.retrievers.impl import BasicRetriever, AdaptiveRetriever
from langchain_ollama import OllamaLLM
from prometheus_fastapi_instrumentator import Instrumentator
from collections import defaultdict 
from langchain_core.documents import Document
from langchain.indexes import SQLRecordManager, index
import shutil
from langchain_community.vectorstores import FAISS
import gc
import time 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Core Service",
    description="A flexible RAG service with multiple strategies and structured error handling.",
    version="1.0"
)

# DB_URL = "postgresql://postgres:mysecretpassword@localhost:5432/rag_db"

Instrumentator().instrument(app).expose(app)
@app.on_event("startup")
async def startup():
    pass

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


def structure_final_response(raw_response: AskResponse) -> StructuredAskResponse:
    answer = raw_response.answer
    
    grouped_chunks = defaultdict(list)
    for doc in raw_response.source_documents:
        metadata = doc.metadata
        doc_uuid = metadata.get("doc_uuid", "unknown_document")
        
        ref_chunk = {
            "content": doc.page_content,
            "page": metadata.get("page", 0)
        }
        grouped_chunks[doc_uuid].append(ref_chunk)
        
   
    references = []
    for doc_uuid, chunks in grouped_chunks.items():
        references.append({
            "doc_uuid": doc_uuid,
            "chunks": chunks
        })
        
    return StructuredAskResponse(answer=answer, references=references)

# --- Service Configuration ---
RETRIEVERS: Dict[str, BaseRetrieverStrategy] = {"basic": BasicRetriever(), "adaptive": AdaptiveRetriever()}
VECTOR_STORE_FACTORY: Dict[str, type[BaseVectorStoreStrategy]] = {"faiss": FAISSStrategy, "chroma": ChromaStrategy}
INDEX_CACHE: Dict[str, BaseVectorStoreStrategy] = {}
llm = OllamaLLM(model=settings.llm.model_name, base_url=settings.services.ollama_base_url)


# --- API Endpoints ---
@app.post("/indexes/{index_name}/add", response_model=CreateSessionResponse, tags=["Indexes"])
def add_to_index(
    index_name: str,
    file: UploadFile = File(...),
    vector_store_strategy: str = Form("faiss", enum=["faiss", "chroma"]),
    extractor_strategy: str = Form(settings.defaults.extractor_strategy),
    chunker_strategy: str = Form(settings.defaults.chunker_strategy)
):
    """یک سند را به ایندکس اضافه کرده و نگاشت شناسه به نام فایل را به‌روز می‌کند."""
    logger.info(f"Request to add document '{file.filename}' to index '{index_name}'...")
    try:
        # --- بخش پردازش و امبدینگ (بدون تغییر) ---
        files = {'file': (file.filename, file.file, file.content_type)}
        params = {'extractor_strategy': extractor_strategy, 'chunker_strategy': chunker_strategy}
        doc_response = requests.post(settings.services.document_processor_url, files=files, data=params, timeout=300)
        doc_response.raise_for_status()
        response_data = doc_response.json()
        chunks = [Chunk(**chunk_data) for chunk_data in response_data['chunks']]
        document_uuid = str(uuid.uuid4())
        for chunk in chunks:
            chunk.metadata['doc_uuid'] = document_uuid
        chunk_texts = [c.chunk_content for c in chunks]
        chunk_metadatas = [c.metadata for c in chunks]
        embed_response = requests.post(settings.services.embedding_service_url, json={"texts": chunk_texts}, timeout=180)
        embed_response.raise_for_status()
        vectors = embed_response.json()["vectors"]
    except requests.RequestException as e:
        raise ServiceException(status_code=503, error_code=40001, message=f"An external service is unavailable: {e}")
    if not vectors:
        raise ServiceException(status_code=400, error_code=30003, message=ERROR_CODES[30003])

    try:
        # --- بخش مدیریت ایندکس و فایل نگاشت ---
        index_dir = os.path.join(settings.paths.index_dir, index_name)
        config_path = os.path.join(index_dir, "index_config.json")
        index_store_path = os.path.join(index_dir, "index")
        
        # ✅ ۱. مسیر فایل نگاشت را تعریف کرده و یک دیکشنری خالی برای آن می‌سازیم
        uuid_map_path = os.path.join(index_dir, "uuid_map.json")
        uuid_map = {}

        if os.path.exists(index_dir):
            logger.info(f"Index '{index_name}' exists. Adding new documents.")
            
            # ✅ ۲. اگر ایندکس وجود داشت، فایل نگاشت قبلی را می‌خوانیم
            if os.path.exists(uuid_map_path):
                with open(uuid_map_path, 'r', encoding='utf-8') as f:
                    uuid_map = json.load(f)

            with open(config_path, 'r', encoding='utf-8') as f: config = json.load(f)
            strategy_class = VECTOR_STORE_FACTORY[config["vector_store_strategy"]]
            index_instance = strategy_class()
            index_instance.load_local(index_store_path)
            index_instance.add_documents(texts=chunk_texts, vectors=vectors, metadatas=chunk_metadatas)
            if isinstance(index_instance.vectorstore, FAISS):
                index_instance.save_local(index_store_path)
            message = f"Document successfully added to index '{index_name}'."
        else:
            logger.info(f"Index '{index_name}' not found. Creating a new one.")
            os.makedirs(index_dir, exist_ok=True)
            strategy_class = VECTOR_STORE_FACTORY[vector_store_strategy]
            index_instance = strategy_class()
            index_instance.create_index(texts=chunk_texts, vectors=vectors, metadatas=chunk_metadatas)
            index_instance.save_local(index_store_path)
            with open(config_path, 'w', encoding='utf-8') as f: json.dump({"vector_store_strategy": vector_store_strategy}, f)
            message = f"New index '{index_name}' created successfully."

        # ✅ ۳. اطلاعات سند جدید را به دیکشنری نگاشت اضافه می‌کنیم
        uuid_map[document_uuid] = file.filename

        # ✅ ۴. دیکشنری به‌روز شده را در فایل uuid_map.json ذخیره می‌کنیم
        with open(uuid_map_path, 'w', encoding='utf-8') as f:
            json.dump(uuid_map, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Updated UUID map for index '{index_name}'.")

        INDEX_CACHE[index_name] = index_instance
        return CreateSessionResponse(index_name=index_name, doc_uuid=document_uuid, message=message, total_chunks=len(vectors))
        
    except Exception as e:
        logger.error(f"Failed during index operation for '{index_name}': {e}", exc_info=True)
        raise ServiceException(status_code=500, error_code=40002, message=f"Failed to create or update index '{index_name}': {e}")

# app/main.py

@app.put("/indexes/{index_name}/documents", tags=["Indexes"])
def update_document(
    index_name: str,
    doc_uuid_to_replace: str = Form(...),
    file: UploadFile = File(...)
):
    """
    یک سند را در ایندکس با جایگزین کردن آن به‌روزرسانی کرده و نام فایل را نیز آپدیت می‌کند.
    """
    logger.info(f"Request to update document '{doc_uuid_to_replace}' in index '{index_name}' with new file '{file.filename}'...")
    
    index_dir = os.path.join(settings.paths.index_dir, index_name)
    config_path = os.path.join(index_dir, "index_config.json")
    index_store_path = os.path.join(index_dir, "index")
    uuid_map_path = os.path.join(index_dir, "uuid_map.json")

    if not os.path.exists(index_dir):
        raise ServiceException(status_code=404, error_code=30002, message=f"Index '{index_name}' not found.")

    # ۱. خواندن تنظیمات و فایل نگاشت
    with open(config_path, 'r', encoding='utf-8') as f: config = json.load(f)
    strategy_name = config["vector_store_strategy"]
    
    uuid_map = {}
    if os.path.exists(uuid_map_path):
        with open(uuid_map_path, 'r', encoding='utf-8') as f:
            uuid_map = json.load(f)
    
    if doc_uuid_to_replace not in uuid_map:
        raise ServiceException(status_code=404, error_code=30002, message=f"Document UUID '{doc_uuid_to_replace}' not found in index '{index_name}'.")

    # ✅ --- شروع بخش کلیدی ---
    # ۲. پردازش و امبدینگ فایل جدید (این بخش اضافه شده است)
    try:
        logger.info(f"Processing new file '{file.filename}'...")
        files = {'file': (file.filename, file.file, file.content_type)}
        params = {'extractor_strategy': settings.defaults.extractor_strategy, 'chunker_strategy': settings.defaults.chunker_strategy}
        
        doc_response = requests.post(settings.services.document_processor_url, files=files, data=params, timeout=300)
        doc_response.raise_for_status()
        response_data = doc_response.json()
        chunks = [Chunk(**chunk_data) for chunk_data in response_data['chunks']]
        
        # شناسه سند قدیمی را به چانک‌های جدید اختصاص می‌دهیم
        for chunk in chunks:
            chunk.metadata['doc_uuid'] = doc_uuid_to_replace
            
        chunk_texts = [c.chunk_content for c in chunks]
        chunk_metadatas = [c.metadata for c in chunks]

        embed_response = requests.post(settings.services.embedding_service_url, json={"texts": chunk_texts}, timeout=180)
        embed_response.raise_for_status()
        vectors = embed_response.json()["vectors"]
        logger.info(f"New file processed into {len(vectors)} chunks.")
    except requests.RequestException as e:
        raise ServiceException(status_code=503, error_code=40001, message=f"An external service is unavailable during update: {e}")
    # ✅ --- پایان بخش کلیدی ---

    # ۳. بارگذاری ایندکس و اجرای منطق به‌روزرسانی
    strategy_class = VECTOR_STORE_FACTORY[strategy_name]
    index_instance = strategy_class()
    index_instance.load_local(index_store_path)

    if strategy_name == 'chroma':
        logger.info("Executing Chroma update strategy (delete and re-add)...")
        index_instance.delete([doc_uuid_to_replace])
        index_instance.add_documents(texts=chunk_texts, vectors=vectors, metadatas=chunk_metadatas)
    
    elif strategy_name == 'faiss':
        logger.info("Executing FAISS update strategy (rebuild)...")
        vector_store = index_instance.vectorstore
        total_docs = vector_store.index.ntotal
        all_ids = list(range(total_docs))
        
        retained_texts, retained_embeddings, retained_metadatas = [], [], []
        
        for i in all_ids:
            doc = vector_store.docstore.search(vector_store.index_to_docstore_id[i])
            if doc.metadata.get("doc_uuid") != doc_uuid_to_replace:
                retained_texts.append(doc.page_content)
                retained_embeddings.append(vector_store.index.reconstruct(i).tolist())
                retained_metadatas.append(doc.metadata)

        final_texts = retained_texts + chunk_texts
        final_vectors = retained_embeddings + vectors
        final_metadatas = retained_metadatas + chunk_metadatas

        new_strategy = FAISSStrategy()
        new_strategy.create_index(texts=final_texts, vectors=final_vectors, metadatas=final_metadatas)
        new_strategy.save_local(index_store_path)
        index_instance = new_strategy

    # ۴. به‌روزرسانی نام فایل در فایل نگاشت
    uuid_map[doc_uuid_to_replace] = file.filename
    with open(uuid_map_path, 'w', encoding='utf-8') as f:
        json.dump(uuid_map, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Updated UUID map with new filename: {file.filename}")

    # ۵. به‌روزرسانی کش حافظه
    INDEX_CACHE[index_name] = index_instance

    return {"message": f"Document '{doc_uuid_to_replace}' was successfully updated in index '{index_name}' with file '{file.filename}'."}
@app.delete("/indexes/{index_name}", status_code=status.HTTP_204_NO_CONTENT, tags=["Indexes"])
def delete_index(index_name: str):
    logger.info(f"Request to delete index '{index_name}'...")
    index_dir = os.path.join(settings.paths.index_dir, index_name)
    if not os.path.exists(index_dir):
        raise ServiceException(status_code=404, error_code=30002, message=f"Index '{index_name}' not found.")
    if index_name in INDEX_CACHE:
        # با None کردن، به پایتون کمک می‌کنیم سریع‌تر منبع را آزاد کند
        INDEX_CACHE[index_name] = None 
        del INDEX_CACHE[index_name]
        logger.info(f"Removed '{index_name}' from in-memory cache.")
        gc.collect()
    max_retries = 5
    retry_delay = 0.5 
    for attempt in range(max_retries):
        try:
            shutil.rmtree(index_dir)
            logger.info(f"Successfully deleted index directory: {index_dir}")
            return  # Exit the function on success
        except OSError as e:
            # Check if the error is the specific file lock error
            if e.winerror == 32 and attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1}: Could not delete index '{index_name}' due to a file lock. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                # If it's another error or the last attempt, raise it
                logger.error(f"Error deleting index directory {index_dir}: {e}", exc_info=True)
                raise ServiceException(status_code=500, error_code=99999, message=f"Failed to delete index files: {e}")
    return




@app.get("/indexes/{index_name}/documents", tags=["Indexes"])
def get_documents_in_index(index_name: str):
    """
    لیست تمام اسناد (نگاشت بین شناسه و نام فایل) را برای یک ایندکس برمی‌گرداند.
    """
    index_dir = os.path.join(settings.paths.index_dir, index_name)
    uuid_map_path = os.path.join(index_dir, "uuid_map.json")

    if not os.path.exists(uuid_map_path):
        raise ServiceException(status_code=404, error_code=30002, message=f"Index '{index_name}' or its document map not found.")

    with open(uuid_map_path, 'r', encoding='utf-8') as f:
        uuid_map = json.load(f)
    
    return uuid_map


@app.post("/indexes/{index_name}/ask", response_model=StructuredAskResponse, tags=["Indexes"])
def ask_from_index(index_name: str, request: AskRequest):
    """Asks a question from a named index."""
    logger.info(f"API request for index '{index_name}' with strategy '{request.retrieval_strategy}'")
    start_time = time.time()
    
    index_instance = INDEX_CACHE.get(index_name)
    if not index_instance:
        logger.info(f"Index not in cache. Loading from disk for index: {index_name}")
        try:
            index_dir = os.path.join(settings.paths.index_dir, index_name)
            config_path = os.path.join(index_dir, "index_config.json")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            strategy_class = VECTOR_STORE_FACTORY[config["vector_store_strategy"]]
            index_instance = strategy_class()
            index_instance.load_local(os.path.join(index_dir, "index"))
            INDEX_CACHE[index_name] = index_instance
        except FileNotFoundError:
            raise ServiceException(status_code=404, error_code=30002, message=f"Index '{index_name}' not found on disk.")
        except Exception as e:
            raise ServiceException(status_code=500, error_code=40003, message=f"Failed to load index '{index_name}': {e}")
            
    retriever = RETRIEVERS.get(request.retrieval_strategy)
    try:
        raw_response: AskResponse = retriever.retrieve(query=request.query, vector_store=index_instance.vectorstore, llm=llm, top_k=request.top_k)
        end_time = time.time()
        
        monitoring_data = {
            "session_id": index_name,
            "query": request.query,
            "retrieval_strategy": request.retrieval_strategy,
            "top_k": request.top_k,
            "llm_answer": raw_response.answer,
            "structured_response ": structure_final_response(raw_response),
            "retrieved_sources_count": len(raw_response.source_documents),
            "source_pages": sorted(list({doc.metadata.get("page") for doc in raw_response.source_documents if doc.metadata.get("page") is not None})),
            "response_time_seconds": round(end_time - start_time, 2),
        }
        monitoring_logger.info("RAG Request Processed", extra=monitoring_data)
        
        structured_response = structure_final_response(raw_response)
        return structured_response
        
    except Exception as e:
        logger.error(f"Error during retrieval for index '{index_name}': {e}", exc_info=True)
        monitoring_logger.error("RAG Request Failed", extra={"session_id": index_name, "query": request.query, "error": str(e)})
        raise ServiceException(status_code=500, error_code=40004, message=f"An error occurred during retrieval/generation: {e}")