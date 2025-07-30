# rag_core_service/app/main.py

import logging
import uuid
import os
import json
from typing import Dict, Set, List
from .monitoring import monitoring_logger
import requests
from fastapi import FastAPI, UploadFile, File, Form, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import time
from .errors import ServiceException, ERROR_CODES
from .core.config import settings
from .models.schemas import (
    CreateSessionResponse, AskRequest, AskResponse, StructuredAskResponse, Chunk,
    SessionInfoResponse, DocumentInfo, AddDocumentResponse, Reference
)
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
from datetime import datetime, timezone
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

def _process_and_embed_file(file: UploadFile, extractor: str, chunker: str, doc_uuid: str) -> (List[str], List[dict], List[List[float]]):
    """فایل را از طریق سرویس‌های خارجی پردازش و امبد می‌کند."""
    try:
        files = {'file': (file.filename, file.file, file.content_type)}
        params = {'extractor_strategy': extractor, 'chunker_strategy': chunker}
        doc_response = requests.post(settings.services.document_processor_url, files=files, data=params, timeout=300)
        doc_response.raise_for_status()
        
        chunks = [Chunk(**chunk_data) for chunk_data in doc_response.json()['chunks']]
        for chunk in chunks:
            chunk.metadata['doc_uuid'] = doc_uuid

        texts = [c.chunk_content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        embed_response = requests.post(settings.services.embedding_service_url, json={"texts": texts}, timeout=180)
        embed_response.raise_for_status()
        vectors = embed_response.json()["vectors"]
        
        if not vectors:
            raise ServiceException(status_code=400, error_code=30003, message=ERROR_CODES[30003])
            
        return texts, metadatas, vectors
    except requests.RequestException as e:
        raise ServiceException(status_code=503, error_code=40001, message=f"یک سرویس خارجی در دسترس نیست: {e}")
    

def structure_final_response(raw_response: AskResponse, session_info: dict) -> StructuredAskResponse:
    """پاسخ خام RAG را به یک ساختار JSON تمیز تبدیل می‌کند."""
    answer = raw_response.answer
    
    doc_map = {doc['doc_uuid']: doc['filename'] for doc in session_info.get('documents', [])}
    
    grouped_chunks = defaultdict(list)
    for doc in raw_response.source_documents:
        metadata = doc.metadata
        doc_uuid = metadata.get("doc_uuid", "unknown_document")
        ref_chunk = {"content": doc.page_content, "page": metadata.get("page", 0)}
        grouped_chunks[doc_uuid].append(ref_chunk)
        
    references = []
    for doc_uuid, chunks in grouped_chunks.items():
        references.append({
            "doc_uuid": doc_uuid,
            "filename": doc_map.get(doc_uuid, "نام فایل نامشخص"),
            "chunks": chunks
        })
        
    return StructuredAskResponse(answer=answer, references=references)

# --- Service Configuration ---
RETRIEVERS: Dict[str, BaseRetrieverStrategy] = {"basic": BasicRetriever(), "adaptive": AdaptiveRetriever()}
VECTOR_STORE_FACTORY: Dict[str, type[BaseVectorStoreStrategy]] = {"faiss": FAISSStrategy, "chroma": ChromaStrategy}
INDEX_CACHE: Dict[str, BaseVectorStoreStrategy] = {}
llm = OllamaLLM(model=settings.llm.model_name, base_url=settings.services.ollama_base_url)



# --- API Endpoints ---

@app.post("/sessions", response_model=CreateSessionResponse, tags=["Sessions"])
def create_session(
    file: UploadFile = File(...),
    vector_store_strategy: str = Form("faiss", enum=["faiss", "chroma"]),
    extractor_strategy: str = Form(settings.defaults.extractor_strategy),
    chunker_strategy: str = Form(settings.defaults.chunker_strategy)
):
    """یک جلسه جدید با اولین سند ایجاد می‌کند."""
    session_id = str(uuid.uuid4())
    doc_uuid = str(uuid.uuid4())
    logger.info(f"درخواست برای ایجاد جلسه جدید '{session_id}' با سند '{file.filename}'...")

    texts, metadatas, vectors = _process_and_embed_file(file, extractor_strategy, chunker_strategy, doc_uuid)

    try:
        session_dir = os.path.join(settings.paths.index_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        strategy_class = VECTOR_STORE_FACTORY[vector_store_strategy]
        index_instance = strategy_class()
        index_instance.create_index(texts=texts, vectors=vectors, metadatas=metadatas)
        index_instance.save_local(os.path.join(session_dir, "index"))

        session_info = {
            "session_id": session_id,
            "vector_store_strategy": vector_store_strategy,
            "documents": [{
                "doc_uuid": doc_uuid,
                "filename": file.filename,
                "added_at": datetime.now(timezone.utc).isoformat()
            }]
        }
        with open(os.path.join(session_dir, "session_info.json"), 'w', encoding='utf-8') as f:
            json.dump(session_info, f, indent=4)

        INDEX_CACHE[session_id] = index_instance
        return CreateSessionResponse(session_id=session_id, doc_uuid=doc_uuid, message="جلسه جدید با موفقیت ایجاد شد.")
    except Exception as e:
        logger.error(f"خطا در ایجاد جلسه '{session_id}': {e}", exc_info=True)
        raise ServiceException(status_code=500, error_code=40002, message=f"خطا در ایجاد جلسه: {e}")

@app.post("/sessions/{session_id}/documents", response_model=AddDocumentResponse, tags=["Sessions"])
def add_document_to_session(session_id: str, file: UploadFile = File(...)):
    """یک سند جدید را به یک جلسه موجود اضافه می‌کند."""
    logger.info(f"درخواست برای افزودن سند '{file.filename}' به جلسه '{session_id}'...")
    session_dir = os.path.join(settings.paths.index_dir, session_id)
    info_path = os.path.join(session_dir, "session_info.json")
    if not os.path.exists(info_path):
        raise ServiceException(status_code=404, error_code=30002, message="جلسه یافت نشد.")

    with open(info_path, 'r', encoding='utf-8') as f:
        session_info = json.load(f)
    
    doc_uuid = str(uuid.uuid4())
    texts, metadatas, vectors = _process_and_embed_file(file, settings.defaults.extractor_strategy, settings.defaults.chunker_strategy, doc_uuid)
    
    strategy_class = VECTOR_STORE_FACTORY[session_info["vector_store_strategy"]]
    index_instance = strategy_class()
    index_instance.load_local(os.path.join(session_dir, "index"))
    index_instance.add_documents(texts=texts, vectors=vectors, metadatas=metadatas)
    
    if isinstance(index_instance.vectorstore, FAISS):
        index_instance.save_local(os.path.join(session_dir, "index"))
        
    session_info["documents"].append({
        "doc_uuid": doc_uuid,
        "filename": file.filename,
        "added_at": datetime.now(timezone.utc).isoformat()
    })
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(session_info, f, indent=4)
        
    INDEX_CACHE[session_id] = index_instance
    return AddDocumentResponse(session_id=session_id, doc_uuid=doc_uuid, message="سند با موفقیت اضافه شد.")

@app.get("/sessions/{session_id}", response_model=SessionInfoResponse, tags=["Sessions"])
def get_session_info(session_id: str):
    """اطلاعات کامل یک جلسه و اسناد آن را برمی‌گرداند."""
    session_dir = os.path.join(settings.paths.index_dir, session_id)
    info_path = os.path.join(session_dir, "session_info.json")
    if not os.path.exists(info_path):
        raise ServiceException(status_code=404, error_code=30002, message="جلسه یافت نشد.")
    
    with open(info_path, 'r', encoding='utf-8') as f:
        return json.load(f)

@app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Sessions"])
def delete_session(session_id: str):
    """یک جلسه و تمام داده‌های مرتبط با آن را حذف می‌کند."""
    logger.info(f"درخواست برای حذف جلسه '{session_id}'...")
    session_dir = os.path.join(settings.paths.index_dir, session_id)
    if not os.path.exists(session_dir):
        raise ServiceException(status_code=404, error_code=30002, message="جلسه یافت نشد.")

    if session_id in INDEX_CACHE:
        del INDEX_CACHE[session_id]
        gc.collect()

    try:
        shutil.rmtree(session_dir)
        logger.info(f"پوشه جلسه با موفقیت حذف شد: {session_dir}")
    except OSError as e:
        logger.error(f"خطا در حذف پوشه جلسه {session_dir}: {e}", exc_info=True)
        raise ServiceException(status_code=500, error_code=99999, message=f"خطا در حذف فایل‌های جلسه: {e}")
    return

@app.post("/sessions/{session_id}/ask", response_model=StructuredAskResponse, tags=["Sessions"])
def ask_from_session(session_id: str, request: AskRequest):
    """از یک جلسه مشخص سوال می‌پرسد."""
    start_time = time.time()
    
    session_dir = os.path.join(settings.paths.index_dir, session_id)
    info_path = os.path.join(session_dir, "session_info.json")
    
    index_instance = INDEX_CACHE.get(session_id)
    if not index_instance:
        logger.info(f"ایندکس در کش نیست. در حال بارگذاری از دیسک برای جلسه: {session_id}")
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                session_info = json.load(f)
            strategy_class = VECTOR_STORE_FACTORY[session_info["vector_store_strategy"]]
            index_instance = strategy_class()
            index_instance.load_local(os.path.join(session_dir, "index"))
            INDEX_CACHE[session_id] = index_instance
        except FileNotFoundError:
            raise ServiceException(status_code=404, error_code=30002, message="جلسه یافت نشد.")
        except Exception as e:
            raise ServiceException(status_code=500, error_code=40003, message=f"خطا در بارگذاری ایندکس جلسه: {e}")
    
    retriever = RETRIEVERS.get(request.retrieval_strategy)
    try:
        raw_response: AskResponse = retriever.retrieve(query=request.query, vector_store=index_instance.vectorstore, llm=llm, top_k=request.top_k)
        
        with open(info_path, 'r', encoding='utf-8') as f:
            session_info = json.load(f)
            
        structured_response = structure_final_response(raw_response, session_info)
        
        monitoring_data = {
            "session_id": session_id, "query": request.query, "retrieval_strategy": request.retrieval_strategy,
            "response_time_seconds": round(time.time() - start_time, 2),
            "llm_answer": raw_response.answer, "structured_response": structured_response.model_dump()
        }
        monitoring_logger.info("RAG Request Processed", extra=monitoring_data)

        return structured_response
        
    except Exception as e:
        logger.error(f"Error during retrieval for session '{session_id}': {e}", exc_info=True)
        monitoring_logger.error("RAG Request Failed", extra={"session_id": session_id, "query": request.query, "error": str(e)})
        raise ServiceException(status_code=500, error_code=40004, message=f"خطا در زمان بازیابی/تولید پاسخ: {e}")

@app.delete("/sessions/{session_id}/documents/{doc_uuid}", tags=["Sessions"])
def delete_document_from_session(session_id: str, doc_uuid: str):
    """
    یک سند مشخص را بر اساس شناسه آن (doc_uuid) از یک جلسه حذف می‌کند.
    """
    logger.info(f"Request to delete document '{doc_uuid}' from session '{session_id}'...")
    
    session_dir = os.path.join(settings.paths.index_dir, session_id)
    info_path = os.path.join(session_dir, "session_info.json")
    index_store_path = os.path.join(session_dir, "index")

    if not os.path.exists(info_path):
        raise ServiceException(status_code=404, error_code=30002, message="Session not found.")

    with open(info_path, 'r', encoding='utf-8') as f:
        session_info = json.load(f)
    
    # بررسی وجود سند در اطلاعات جلسه
    doc_to_delete = next((doc for doc in session_info["documents"] if doc["doc_uuid"] == doc_uuid), None)
    if not doc_to_delete:
        raise ServiceException(status_code=404, error_code=30002, message=f"Document UUID '{doc_uuid}' not found in session.")

    strategy_name = session_info["vector_store_strategy"]
    strategy_class = VECTOR_STORE_FACTORY[strategy_name]
    index_instance = strategy_class()
    index_instance.load_local(index_store_path)

    # اجرای منطق حذف بر اساس نوع ایندکس
    if strategy_name == 'chroma':
        index_instance.delete([doc_uuid])
    elif strategy_name == 'faiss':
        # برای FAISS، از روش بازسازی ایندکس استفاده می‌کنیم
        vector_store = index_instance.vectorstore
        retained_texts, retained_vectors, retained_metadatas = [], [], []
        for i in range(vector_store.index.ntotal):
            doc = vector_store.docstore.search(vector_store.index_to_docstore_id[i])
            if doc.metadata.get("doc_uuid") != doc_uuid:
                retained_texts.append(doc.page_content)
                retained_vectors.append(vector_store.index.reconstruct(i).tolist())
                retained_metadatas.append(doc.metadata)
        
        if not retained_texts:
            # اگر هیچ سندی باقی نماند، کل جلسه را حذف می‌کنیم
            shutil.rmtree(session_dir)
            if session_id in INDEX_CACHE: del INDEX_CACHE[session_id]
            return {"message": "Session was empty after deletion and has been removed."}
        
        new_strategy = FAISSStrategy()
        new_strategy.create_index(texts=retained_texts, vectors=retained_vectors, metadatas=retained_metadatas)
        new_strategy.save_local(index_store_path)
        index_instance = new_strategy

    # به‌روزرسانی فایل اطلاعات جلسه
    session_info["documents"] = [doc for doc in session_info["documents"] if doc["doc_uuid"] != doc_uuid]
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(session_info, f, indent=4)
    
    INDEX_CACHE[session_id] = index_instance
    return {"message": f"Document '{doc_uuid}' was successfully deleted from session '{session_id}'."}

# app/main.py

@app.put("/sessions/{session_id}/documents/{doc_uuid}", tags=["Sessions"])
def update_document_in_session(
    session_id: str,
    doc_uuid: str,
    file: UploadFile = File(...)
):
    """
    Atomically updates a document within a session by replacing it with a new file.
    It deletes all old chunks and adds the new ones.
    """
    logger.info(f"Request to update document '{doc_uuid}' in session '{session_id}' with new file '{file.filename}'...")
    
    session_dir = os.path.join(settings.paths.index_dir, session_id)
    info_path = os.path.join(session_dir, "session_info.json")
    index_store_path = os.path.join(session_dir, "index")

    if not os.path.exists(info_path):
        raise ServiceException(status_code=404, error_code=30002, message="Session not found.")

    with open(info_path, 'r', encoding='utf-8') as f:
        session_info = json.load(f)
    
    doc_to_update = next((doc for doc in session_info["documents"] if doc["doc_uuid"] == doc_uuid), None)
    if not doc_to_update:
        raise ServiceException(status_code=404, error_code=30002, message=f"Document UUID '{doc_uuid}' not found in session.")

    # 1. Process and embed the NEW file
    new_texts, new_metadatas, new_vectors = _process_and_embed_file(
        file, settings.defaults.extractor_strategy, settings.defaults.chunker_strategy, doc_uuid
    )
    
    # 2. Load the index
    strategy_name = session_info["vector_store_strategy"]
    strategy_class = VECTOR_STORE_FACTORY[strategy_name]
    index_instance = strategy_class()
    index_instance.load_local(index_store_path)

    # 3. Execute the update logic based on the strategy
    if strategy_name == 'chroma':
        logger.info("Executing Chroma update strategy (delete and re-add)...")
        index_instance.delete([doc_uuid])
        index_instance.add_documents(texts=new_texts, vectors=new_vectors, metadatas=new_metadatas)
    
    elif strategy_name == 'faiss':
        logger.info("Executing FAISS update strategy (rebuild)...")
        vector_store = index_instance.vectorstore
        
        retained_texts, retained_vectors, retained_metadatas = [], [], []
        
        for i in range(vector_store.index.ntotal):
            doc = vector_store.docstore.search(vector_store.index_to_docstore_id[i])
            if doc.metadata.get("doc_uuid") != doc_uuid:
                retained_texts.append(doc.page_content)
                retained_vectors.append(vector_store.index.reconstruct(i).tolist())
                retained_metadatas.append(doc.metadata)

        final_texts = retained_texts + new_texts
        final_vectors = retained_vectors + new_vectors
        final_metadatas = retained_metadatas + new_metadatas

        new_strategy = FAISSStrategy()
        new_strategy.create_index(texts=final_texts, vectors=final_vectors, metadatas=final_metadatas)
        new_strategy.save_local(index_store_path)
        index_instance = new_strategy

    # 4. Update the session_info.json file with the new filename
    for doc in session_info["documents"]:
        if doc["doc_uuid"] == doc_uuid:
            doc["filename"] = file.filename
            doc["added_at"] = datetime.now(timezone.utc).isoformat()
            break
            
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(session_info, f, indent=4)
    
    # 5. Update the in-memory cache
    INDEX_CACHE[session_id] = index_instance

    return {"message": f"Document '{doc_uuid}' was successfully updated with file '{file.filename}'."}