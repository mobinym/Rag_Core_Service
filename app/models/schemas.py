# rag_core_service/app/models/schemas.py

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from app.core.config import settings
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

class FormattedAskResponse(BaseModel):
    formatted_answer: str