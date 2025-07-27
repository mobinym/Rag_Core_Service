# rag_core_service/app/errors.py

from typing import Dict, Any

class ServiceException(Exception):
    """Custom base exception for this service."""
    def __init__(self, status_code: int, error_code: int, message: str, details: Any = None):
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.details = details
        super().__init__(self.message)

ERROR_CODES: Dict[int, str] = {
    30001: "Strategy not supported",
    30002: "Session ID not found",
    30003: "Document processing resulted in no chunks",
    30004: "Session is corrupted or invalid",
    
    40001: "External service is unavailable",
    40002: "Failed to create or save the index",
    40003: "Failed to load the index from disk",
    40004: "An error occurred during the retrieval and generation process",
    
    90001: "Invalid input provided",
    99999: "An unexpected internal server error occurred"
}