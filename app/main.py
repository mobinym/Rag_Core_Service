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
from .models.schemas import CreateSessionResponse, Chunk, AskRequest, AskResponse, FormattedAskResponse
from .strategies.vector_stores.base import BaseVectorStoreStrategy
from .strategies.vector_stores.impl import FAISSStrategy, ChromaStrategy
from .strategies.retrievers.base import BaseRetrieverStrategy
from .strategies.retrievers.impl import BasicRetriever, AdaptiveRetriever
from langchain_ollama import OllamaLLM
from prometheus_fastapi_instrumentator import Instrumentator


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Core Service",
    description="A flexible RAG service with multiple strategies and structured error handling.",
    version="1.0"
)


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

from collections import defaultdict # Make sure this import is at the top of the file

def format_rag_response(raw_response: AskResponse) -> str:
    """
    Converts the raw RAG response to a clean, formatted Markdown text,
    grouping sources by their document ID.
    """
    answer = raw_response.answer
    
    # Use a defaultdict to group pages by their doc_uuid
    sources = defaultdict(set)
    for doc in raw_response.source_documents:
        metadata = doc.metadata
        # Assumes the 'doc_uuid' key exists in the metadata
        doc_uuid = metadata.get("doc_uuid", "Unknown Document")
        page_num = metadata.get("page")
        
        if page_num is not None:
            sources[doc_uuid].add(page_num)
            
    # Build the final string using Markdown format
    formatted_output = f"**پاسخ:**\n\n{answer}"
    
    if sources:
        formatted_output += "\n\n---\n\n**منابع:**\n"
        for doc_uuid, pages in sources.items():
            # Convert the list of pages to a readable string
            page_str = ", ".join(map(str, sorted(list(pages))))
            formatted_output += f"\n* سند: `{doc_uuid}` (صفحات: {page_str})"
            
    return formatted_output

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
    """Adds a document to a named index. If the index doesn't exist, it will be created."""
    logger.info(f"Request to add document to index '{index_name}'...")
    
    try:
        files = {'file': (file.filename, file.file, file.content_type)}
        params = {'extractor_strategy': extractor_strategy, 'chunker_strategy': chunker_strategy}
        doc_response = requests.post(settings.services.document_processor_url, files=files, data=params, timeout=300)
        doc_response.raise_for_status()
        
        response_data = doc_response.json()
        chunks = [Chunk(**chunk_data) for chunk_data in response_data['chunks']]
        document_uuid = str(uuid.uuid4())
        
        # ۲. این UUID را به متادیتای تمام چانک‌ها اضافه می‌کنیم
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
        index_dir = os.path.join(settings.paths.index_dir, index_name)
        config_path = os.path.join(index_dir, "index_config.json")
        index_store_path = os.path.join(index_dir, "index")
        
        if os.path.exists(index_dir):
            logger.info(f"Index '{index_name}' exists. Adding new documents.")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            strategy_class = VECTOR_STORE_FACTORY[config["vector_store_strategy"]]
            index_instance = strategy_class()
            index_instance.load_local(index_store_path)
            index_instance.add_documents(texts=chunk_texts, vectors=vectors, metadatas=chunk_metadatas)
            # index_instance.save_local(index_store_path)
            message = f"Document successfully added to index '{index_name}'."
        else:
            logger.info(f"Index '{index_name}' not found. Creating a new one.")
            os.makedirs(index_dir, exist_ok=True)
            strategy_class = VECTOR_STORE_FACTORY[vector_store_strategy]
            index_instance = strategy_class()
            index_instance.create_index(texts=chunk_texts, vectors=vectors, metadatas=chunk_metadatas)
            index_instance.save_local(index_store_path)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump({"vector_store_strategy": vector_store_strategy}, f)
            message = f"New index '{index_name}' created successfully."

        INDEX_CACHE[index_name] = index_instance
        return CreateSessionResponse(session_id=index_name, message=message, total_chunks=len(vectors))
        
    except Exception as e:
        logger.error(f"Failed during index operation for '{index_name}': {e}", exc_info=True)
        raise ServiceException(status_code=500, error_code=40002, message=f"Failed to create or update index '{index_name}': {e}")


@app.post("/indexes/{index_name}/ask", response_model=FormattedAskResponse, tags=["Indexes"])
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
            "final_formatted_answer": format_rag_response(raw_response),
            "retrieved_sources_count": len(raw_response.source_documents),
            "source_pages": sorted(list({doc.metadata.get("page") for doc in raw_response.source_documents if doc.metadata.get("page") is not None})),
            "response_time_seconds": round(end_time - start_time, 2),
        }
        monitoring_logger.info("RAG Request Processed", extra=monitoring_data)
        
        formatted_string = format_rag_response(raw_response)
        return FormattedAskResponse(formatted_answer=formatted_string)
        
    except Exception as e:
        logger.error(f"Error during retrieval for index '{index_name}': {e}", exc_info=True)
        monitoring_logger.error("RAG Request Failed", extra={"session_id": index_name, "query": request.query, "error": str(e)})
        raise ServiceException(status_code=500, error_code=40004, message=f"An error occurred during retrieval/generation: {e}")