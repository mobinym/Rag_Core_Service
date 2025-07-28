# app/models/schemas.py

from pydantic import BaseModel
from typing import List, Dict, Any
from app.core.config import settings

# --- مدل‌های ورودی و داخلی ---

class Chunk(BaseModel):
    chunk_content: str
    metadata: Dict[str, Any]

class CreateSessionResponse(BaseModel):
    session_id: str
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

# --- ✅ مدل‌های جدید برای خروجی نهایی ---

class ReferenceChunk(BaseModel):
    """
    مدل یک قطعه متن (chunk) که به عنوان منبع استفاده شده.
    """
    content: str
    page: int

class Reference(BaseModel):
    """
    مدل یک سند (فایل) که به عنوان منبع استفاده شده.
    """
    doc_uuid: str
    chunks: List[ReferenceChunk]

class StructuredAskResponse(BaseModel):
    """
    مدل نهایی و ساختاریافته پاسخ که به کاربر بازگردانده می‌شود.
    """
    answer: str
    references: List[Reference]