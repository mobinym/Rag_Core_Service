# rag_core_service/app/models/schemas.py

from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class Chunk(BaseModel):
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
    page_content: str
    metadata: Dict[str, Any]
    score: float

class AskResponse(BaseModel):
    """
    این مدل پاسخ خام و ساختاریافته است که توسط retriever تولید می‌شود.
    """
    answer: str
    source_documents: List[SourceDocument]

# ✅ مدل پاسخ جدید برای خروجی نهایی و تمیز به کاربر
class FormattedAskResponse(BaseModel):
    """
    این مدل پاسخ نهایی است که به صورت فرمت‌شده و خوانا برای کاربر ارسال می‌شود.
    """
    formatted_answer: str