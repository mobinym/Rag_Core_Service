# app/core/config.py

import yaml
from pydantic import BaseModel
from pathlib import Path

# مدل‌های Pydantic برای اعتبارسنجی تنظیمات خوانده شده از فایل YAML
class Paths(BaseModel):
    index_dir: str

class Defaults(BaseModel):
    extractor_strategy: str
    chunker_strategy: str
    retrieval_strategy: str
    top_k: int

class Services(BaseModel):
    document_processor_url: str
    embedding_service_url: str
    ollama_base_url: str

class LLM(BaseModel):
    model_name: str

class AdaptiveRetrieverSettings(BaseModel):
    min_answer_length: int
    retry_k: int

class RetrieverSettings(BaseModel):
    adaptive: AdaptiveRetrieverSettings


class PromptTemplates(BaseModel):
    adaptive_retriever: str


class Settings(BaseModel):
    paths: Paths
    services: Services
    llm: LLM
    defaults: Defaults 
    prompt_templates: PromptTemplates

    retriever_settings: RetrieverSettings


def load_settings() -> Settings:
    config_path = Path(__file__).parent.parent.parent / "config.yml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    
    return Settings(**config_data)


settings = load_settings()