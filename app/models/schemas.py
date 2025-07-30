# app/models/schemas.py

from pydantic import BaseModel
from typing import List, Dict, Any
from app.core.config import settings

# --- مدل‌های ورودی و داخلی ---
class Chunk(BaseModel):
    chunk_content: str
    metadata: Dict[str, Any]

class AskRequest(BaseModel):
    query: str
    retrieval_strategy: str = settings.defaults.retrieval_strategy
    top_k: int = settings.defaults.top_k

class SourceDocument(BaseModel):
    page_content: str
    metadata: Dict[str, Any]
    score: float

class AskResponse(BaseModel):
    answer: str
    source_documents: List[SourceDocument]

# --- مدل‌های خروجی نهایی ---
class DocumentInfo(BaseModel):
    doc_uuid: str
    filename: str
    added_at: str

class CreateSessionResponse(BaseModel):
    session_id: str
    doc_uuid: str
    message: str

class AddDocumentResponse(BaseModel):
    session_id: str
    doc_uuid: str
    message: str

class SessionInfoResponse(BaseModel):
    session_id: str
    vector_store_strategy: str
    documents: List[DocumentInfo]

class ReferenceChunk(BaseModel):
    content: str
    page: int

class Reference(BaseModel):
    doc_uuid: str
    filename: str
    chunks: List[ReferenceChunk]

class StructuredAskResponse(BaseModel):
    answer: str
    references: List[Reference]