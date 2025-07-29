# app/models/schemas.py

from pydantic import BaseModel
from typing import List, Dict, Any
from app.core.config import settings


class Chunk(BaseModel):
    chunk_content: str
    metadata: Dict[str, Any]

# app/models/schemas.py
class CreateSessionResponse(BaseModel):
    index_name: str # ✅ تغییر نام از session_id
    doc_uuid: str   # ✅ افزودن شناسه سند
    message: str
    total_chunks: int

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


class ReferenceChunk(BaseModel):
    content: str
    page: int

class Reference(BaseModel):
    doc_uuid: str
    chunks: List[ReferenceChunk]

class StructuredAskResponse(BaseModel):
    answer: str
    references: List[Reference]