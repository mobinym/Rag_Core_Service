#app/core/config.py
class Settings:
    INDEX_DIR: str = "data/indices" 
    DOCUMENT_PROCESSOR_URL: str = "http://services.aiopt.io:7000/v1/chn/chunking/"
    EMBEDDING_SERVICE_URL: str = "http://services.aiopt.io:5000/v1/emd/embed-text/"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
settings = Settings()