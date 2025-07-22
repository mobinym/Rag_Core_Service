# app/core/config.py
class Settings:
    # آدرس سرویس پردازش اسناد
    DOCUMENT_PROCESSOR_URL: str = "http://localhost:8001/v1/chn/chunking/"
    
    # آدرس سرویس امبدینگ شما که روی سرور قرار دارد
    EMBEDDING_SERVICE_URL: str = "http://localhost:8002/v1/emd/embed-text/"
    
    # آدرس سرور Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"

settings = Settings()