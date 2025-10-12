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
    AskRequest, AskResponse, StructuredAskResponse, Chunk,
    DocumentInfo, AddDocumentResponse, Reference,
    CreateVSResponse, VSInfoResponse, RetrieveResponse 
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

def _process_and_embed_file(file: UploadFile, extractor: str, chunker: str, doc_uuid: str) -> tuple[List[str], List[dict], List[List[float]]]:
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
@app.post("/v1/rag/vs/empty", response_model=CreateVSResponse, tags=["Vector Stores"])
def create_empty_vs(
    vector_store_strategy: str = Form("faiss", enum=["faiss", "chroma"])
):
    """یک Vector Store (VS) جدید و خالی ایجاد کرده و شناسه آن را برمی‌گرداند."""
    vs_id = str(uuid.uuid4())
    logger.info(f"درخواست برای ایجاد VS خالی جدید '{vs_id}' با استراتژی '{vector_store_strategy}'...")

    try:
        vs_dir = os.path.join(settings.paths.index_dir, vs_id)
        os.makedirs(vs_dir, exist_ok=True)
        
        strategy_class = VECTOR_STORE_FACTORY[vector_store_strategy]
        index_instance = strategy_class()

        collection_metadata = None
        if vector_store_strategy == 'chroma':
            collection_metadata = {"hnsw:space": "cosine"}

        index_instance.create_and_save_empty(
            path=os.path.join(vs_dir, "index"),
            metadatas=collection_metadata
        )

        
        vs_info = {
            "vs_id": vs_id,
            "vector_store_strategy": vector_store_strategy,
            "documents": [] 
        }
        with open(os.path.join(vs_dir, "vs_info.json"), 'w', encoding='utf-8') as f:
            json.dump(vs_info, f, indent=4, ensure_ascii=False)

        return CreateVSResponse(
            vs_id=vs_id,
            doc_uuid="", 
            message="VS خالی با موفقیت ایجاد شد. اکنون می‌توانید اسناد را به آن اضافه کنید."
        )
    except Exception as e:
        logger.error(f"خطا در ایجاد VS خالی '{vs_id}': {e}", exc_info=True)
        raise ServiceException(status_code=500, error_code=40002, message=f"خطا در ایجاد VS خالی: {e}")

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.post("/v1/rag/vs/{vs_id}/documents", response_model=AddDocumentResponse, tags=["Vector Stores"])
def add_document_to_vs(
    vs_id: str,
    file: UploadFile = File(...),
    
    extractor_strategy: str = Form(settings.defaults.extractor_strategy),
    chunker_strategy: str = Form(settings.defaults.chunker_strategy)
):
    """یک سند جدید را با استراتژی‌های پردازش مشخص شده به یک VS موجود اضافه می‌کند."""
    logger.info(f"درخواست برای افزودن سند '{file.filename}' به VS '{vs_id}'...")
    
    vs_dir = os.path.join(settings.paths.index_dir, vs_id)
    info_path = os.path.join(vs_dir, "vs_info.json")
    index_store_path = os.path.join(vs_dir, "index")
    

    if not os.path.exists(info_path):
        raise ServiceException(status_code=404, error_code=30002, message="VS یافت نشد. لطفاً ابتدا یک VS خالی بسازید.")


    doc_uuid = str(uuid.uuid4())
    texts, metadatas, vectors = _process_and_embed_file(
        file, 
        extractor_strategy, 
        chunker_strategy,   
        doc_uuid
    )
    
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            vs_info = json.load(f)
        
        strategy_class = VECTOR_STORE_FACTORY[vs_info["vector_store_strategy"]]
        index_instance = strategy_class()
        index_instance.load_local(index_store_path)
        index_instance.add_documents(texts=texts, vectors=vectors, metadatas=metadatas)
        
        if isinstance(index_instance.vectorstore, FAISS):
            index_instance.save_local(index_store_path)
        
       
        vs_info["documents"].append({
            "doc_uuid": doc_uuid,
            "filename": file.filename,
            "added_at": datetime.now(timezone.utc).isoformat()
        })
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(vs_info, f, indent=4, ensure_ascii=False)
        
        
        INDEX_CACHE[vs_id] = index_instance
        return AddDocumentResponse(vs_id=vs_id, doc_uuid=doc_uuid, message="سند با موفقیت به VS موجود اضافه شد.")

    except Exception as e:
        logger.error(f"خطا در افزودن سند به VS '{vs_id}': {e}", exc_info=True)
        raise ServiceException(status_code=500, error_code=40002, message=f"خطا در عملیات VS: {e}")
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

@app.get("/v1/rag/vs/{vs_id}", response_model=VSInfoResponse, tags=["Vector Stores"])
def get_vs_info(vs_id: str):
    """اطلاعات کامل یک VS (Vector Store) و اسناد آن را برمی‌گرداند."""
    vs_dir = os.path.join(settings.paths.index_dir, vs_id)
    info_path = os.path.join(vs_dir, "vs_info.json") 
    
    if not os.path.exists(info_path):
        raise ServiceException(status_code=404, error_code=30002, message="VS یافت نشد.")
        
    with open(info_path, 'r', encoding='utf-8') as f:
        return json.load(f)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.post("/v1/rag/vs/{vs_id}/chat", response_model=StructuredAskResponse, tags=["Vector Stores"])
def ask_from_vs(vs_id: str, request: AskRequest):
    """Asks a question from a specific VS (Vector Store)."""
    start_time = time.time()
    logger.info(f"API request for VS '{vs_id}' with strategy '{request.retrieval_strategy}'")
    
    vs_dir = os.path.join(settings.paths.index_dir, vs_id)
    info_path = os.path.join(vs_dir, "vs_info.json")
    
    index_instance = INDEX_CACHE.get(vs_id)
    if not index_instance:
        logger.info(f"Index not in cache. Loading from disk for VS: {vs_id}")
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                vs_info = json.load(f)
            strategy_class = VECTOR_STORE_FACTORY[vs_info["vector_store_strategy"]]
            index_instance = strategy_class()
            index_instance.load_local(os.path.join(vs_dir, "index"))
            INDEX_CACHE[vs_id] = index_instance
        except FileNotFoundError:
            raise ServiceException(status_code=404, error_code=30002, message="VS not found.")
        except Exception as e:
            raise ServiceException(status_code=500, error_code=40003, message=f"Failed to load VS index: {e}")
    
    retriever = RETRIEVERS.get(request.retrieval_strategy)
    try:
        raw_response: AskResponse = retriever.retrieve(query=request.query, vector_store=index_instance.vectorstore, llm=llm, top_k=request.top_k)
        
        with open(info_path, 'r', encoding='utf-8') as f:
            vs_info = json.load(f)
            
        structured_response = structure_final_response(raw_response, vs_info)
        
        # (Monitoring log section)
        monitoring_data = {
            "session_id": vs_id, # Keeping "session_id" for internal log consistency is fine
            "query": request.query,
            "retrieval_strategy": request.retrieval_strategy,
            "response_time_seconds": round(time.time() - start_time, 2),
            "llm_answer": raw_response.answer,
            "structured_response": structured_response.model_dump()
        }
        monitoring_logger.info("RAG Request Processed", extra=monitoring_data)

        return structured_response
        
    except Exception as e:
        logger.error(f"Error during retrieval for VS '{vs_id}': {e}", exc_info=True)
        monitoring_logger.error("RAG Request Failed", extra={"session_id": vs_id, "query": request.query, "error": str(e)})
        raise ServiceException(status_code=500, error_code=40004, message=f"An error occurred during retrieval/generation: {e}")
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.post("/v1/rag/vs/{vs_id}/retrieve", response_model=RetrieveResponse, tags=["Vector Stores"])
def retrieve_from_vs(vs_id: str, request: AskRequest):
    """
    Retrieves relevant source documents from a specific VS without generating an answer.
    This is ideal for scenarios where the final answer generation is handled by another service (e.g., a Chainlit UI).
    """
    logger.info(f"Retrieval-only request for VS '{vs_id}' with strategy '{request.retrieval_strategy}'")
    
    vs_dir = os.path.join(settings.paths.index_dir, vs_id)
    
    
    index_instance = INDEX_CACHE.get(vs_id)
    if not index_instance:
        logger.info(f"Index not in cache. Loading from disk for VS: {vs_id}")
        try:
            with open(os.path.join(vs_dir, "vs_info.json"), 'r', encoding='utf-8') as f:
                vs_info = json.load(f)
            strategy_class = VECTOR_STORE_FACTORY[vs_info["vector_store_strategy"]]
            index_instance = strategy_class()
            index_instance.load_local(os.path.join(vs_dir, "index"))
            INDEX_CACHE[vs_id] = index_instance
        except FileNotFoundError:
            raise ServiceException(status_code=404, error_code=30002, message="VS not found.")
        except Exception as e:
            raise ServiceException(status_code=500, error_code=40003, message=f"Failed to load VS index: {e}")
    
    retriever = RETRIEVERS.get(request.retrieval_strategy)
    if not retriever:
        raise ServiceException(status_code=400, error_code=30001, message=f"Retrieval strategy '{request.retrieval_strategy}' not supported.")

    try:
        response = retriever.retrieve_documents(
            query=request.query, 
            vector_store=index_instance.vectorstore, 
            top_k=request.top_k
        )
        return response
        
    except Exception as e:
        logger.error(f"Error during document retrieval for VS '{vs_id}': {e}", exc_info=True)
        raise ServiceException(status_code=500, error_code=40004, message=f"An error occurred during retrieval: {e}")

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.delete("/v1/rag/vs/{vs_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Vector Stores"])
def delete_vs(vs_id: str):
    """یک VS (Vector Store) و تمام داده‌های مرتبط با آن را حذف می‌کند."""
    logger.info(f"درخواست برای حذف VS '{vs_id}'...")
    
    vs_dir = os.path.join(settings.paths.index_dir, vs_id)
    
    if not os.path.exists(vs_dir):
        raise ServiceException(status_code=404, error_code=30002, message="VS یافت نشد.")
        
    if vs_id in INDEX_CACHE:
        del INDEX_CACHE[vs_id]
        gc.collect()
        
    try:
        shutil.rmtree(vs_dir)
        logger.info(f"پوشه VS با موفقیت حذف شد: {vs_dir}")
    except OSError as e:
        logger.error(f"خطا در حذف پوشه VS '{vs_dir}': {e}", exc_info=True)
        raise ServiceException(status_code=500, error_code=99999, message=f"خطا در حذف فایل‌های VS: {e}")
        
    return
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

@app.delete("/v1/rag/vs/{vs_id}/documents/{doc_uuid}", tags=["Vector Stores"])
def delete_document_from_vs(vs_id: str, doc_uuid: str):
    """
    یک سند مشخص را بر اساس شناسه آن (doc_uuid) از یک VS حذف می‌کند.
    """
    logger.info(f"درخواست برای حذف سند '{doc_uuid}' از VS '{vs_id}'...")
    
    vs_dir = os.path.join(settings.paths.index_dir, vs_id)
    info_path = os.path.join(vs_dir, "vs_info.json")
    index_store_path = os.path.join(vs_dir, "index")

    if not os.path.exists(info_path):
        raise ServiceException(status_code=404, error_code=30002, message="VS یافت نشد.")

    with open(info_path, 'r', encoding='utf-8') as f:
        vs_info = json.load(f)
    
    doc_to_delete = next((doc for doc in vs_info["documents"] if doc["doc_uuid"] == doc_uuid), None)
    if not doc_to_delete:
        raise ServiceException(status_code=404, error_code=30002, message=f"شناسه سند '{doc_uuid}' در VS یافت نشد.")

    strategy_name = vs_info["vector_store_strategy"]
    strategy_class = VECTOR_STORE_FACTORY[strategy_name]
    index_instance = strategy_class()
    index_instance.load_local(index_store_path)


    if strategy_name == 'chroma':
        logger.info("در حال اجرای استراتژی حذف برای Chroma...")
        index_instance.delete([doc_uuid])
        message = f"سند '{doc_uuid}' با موفقیت از ایندکس Chroma حذف شد."
    
    elif strategy_name == 'faiss':
        logger.info("در حال اجرای استراتژی حذف برای FAISS (بازسازی)...")
        vector_store = index_instance.vectorstore
        
        retained_texts, retained_vectors, retained_metadatas = [], [], []
        
        
        for i in range(vector_store.index.ntotal):
            doc = vector_store.docstore.search(vector_store.index_to_docstore_id[i])
            if doc.metadata.get("doc_uuid") != doc_uuid:
                retained_texts.append(doc.page_content)
                retained_vectors.append(vector_store.index.reconstruct(i).tolist())
                retained_metadatas.append(doc.metadata)
        
        if not retained_texts:
       
            shutil.rmtree(vs_dir)
            if vs_id in INDEX_CACHE: del INDEX_CACHE[vs_id]
            return {"message": f"VS '{vs_id}' پس از حذف سند خالی شد و به طور کامل حذف گردید."}
        
     
        new_strategy = FAISSStrategy()
        new_strategy.create_index(texts=retained_texts, vectors=retained_vectors, metadatas=retained_metadatas)
        new_strategy.save_local(index_store_path)
        index_instance = new_strategy
        message = f"سند '{doc_uuid}' با بازسازی ایندکس FAISS حذف شد."

   
    original_filename = doc_to_delete.get("filename", "N/A")
    vs_info["documents"] = [doc for doc in vs_info["documents"] if doc["doc_uuid"] != doc_uuid]
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(vs_info, f, indent=4, ensure_ascii=False)
    
    logger.info(f"سند '{original_filename}' (UUID: {doc_uuid}) از فایل اطلاعات حذف شد.")

  
    INDEX_CACHE[vs_id] = index_instance

    return {"message": message}
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


@app.put("/v1/rag/vs/{vs_id}/documents/{doc_uuid}", tags=["Vector Stores"])
def update_document_in_vs(
    vs_id: str,
    doc_uuid: str,
    file: UploadFile = File(...)
):
    """
    Atomically updates a document within a VS by replacing it with a new file.
    """
    logger.info(f"Request to update document '{doc_uuid}' in VS '{vs_id}' with new file '{file.filename}'...")
    
    vs_dir = os.path.join(settings.paths.index_dir, vs_id)
    info_path = os.path.join(vs_dir, "vs_info.json")
    index_store_path = os.path.join(vs_dir, "index")

    if not os.path.exists(info_path):
        raise ServiceException(status_code=404, error_code=30002, message="VS not found.")

    with open(info_path, 'r', encoding='utf-8') as f:
        vs_info = json.load(f)
    
    doc_to_update = next((doc for doc in vs_info["documents"] if doc["doc_uuid"] == doc_uuid), None)
    if not doc_to_update:
        raise ServiceException(status_code=404, error_code=30002, message=f"Document UUID '{doc_uuid}' not found in VS.")

    new_texts, new_metadatas, new_vectors = _process_and_embed_file(
        file, settings.defaults.extractor_strategy, settings.defaults.chunker_strategy, doc_uuid
    )
    
    strategy_name = vs_info["vector_store_strategy"]
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

    # 4. Update the vs_info.json file with the new filename
    for doc in vs_info["documents"]:
        if doc["doc_uuid"] == doc_uuid:
            doc["filename"] = file.filename
            doc["added_at"] = datetime.now(timezone.utc).isoformat()
            break
            
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(vs_info, f, indent=4, ensure_ascii=False)
    
    # 5. Update the in-memory cache
    INDEX_CACHE[vs_id] = index_instance

    return {"message": f"Document '{doc_uuid}' was successfully updated with file '{file.filename}'."}
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------