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

# مدل اصلی تنظیمات که تمام بخش‌ها را شامل می‌شود
class Settings(BaseModel):
    paths: Paths
    services: Services
    llm: LLM
    defaults: Defaults # ✅ اضافه کردن مدل پیش‌فرض‌ها
    retriever_settings: RetrieverSettings


def load_settings() -> Settings:
    """
    فایل config.yml را خوانده، اعتبارسنجی کرده و یک آبجکت Settings برمی‌گرداند.
    """
    # پیدا کردن مسیر فایل کانفیگ در ریشه پروژه
    config_path = Path(__file__).parent.parent.parent / "config.yml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    
    # اعتبارسنجی و ساخت آبجکت تنظیمات با Pydantic
    return Settings(**config_data)

# ساخت یک نمونه از تنظیمات برای استفاده در کل برنامه
settings = load_settings()