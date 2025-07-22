# app/models/schemas.py
from pydantic import BaseModel
from typing import List, Dict, Any

# --- مدل‌های مربوط به ایجاد جلسه ---
class Chunk(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

class CreateSessionResponse(BaseModel):
    session_id: str
    message: str
    total_chunks: int

# --- ✅ مدل‌های جدید برای پرسش و پاسخ ---
class AskRequest(BaseModel):
    query: str
    # در آینده استراتژی‌های پیشرفته را اینجا اضافه خواهیم کرد
    # retrieval_strategy: str = "basic" 
    top_k: int = 3

class SourceDocument(BaseModel):
    page_content: str
    metadata: Dict[str, Any]
    score: float # امتیاز شباهت را هم اضافه می‌کنیم

class AskResponse(BaseModel):
    answer: str
    source_documents: List[SourceDocument]