# rag_core_service/app/models/schemas.py
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class Chunk(BaseModel):
    # ✅ نام فیلد مطابق با خروجی سرویس document_processor اصلاح شد
    chunk_content: str
    metadata: Dict[str, Any]

class CreateSessionResponse(BaseModel):
    session_id: str
    message: str
    total_chunks: int

class AskRequest(BaseModel):
    query: str
    retrieval_strategy: str = "adaptive"
    top_k: int = 3

class SourceDocument(BaseModel):
    # این نام استاندارد لانگ‌چین است و بهتر است همین باقی بماند
    page_content: str
    metadata: Dict[str, Any]
    score: float

class AskResponse(BaseModel):
    answer: str
    source_documents: List[SourceDocument]