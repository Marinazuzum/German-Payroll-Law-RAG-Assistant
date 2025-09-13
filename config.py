import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings."""
    
    # API Keys
    openai_api_key: str = ""
    
    # Paths
    chroma_persist_directory: str = "./data/chroma_db"
    pdf_data_path: str = "./data/raw"
    processed_data_path: str = "./data/processed"
    
    # Chunking parameters
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Model settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    max_tokens: int = 1000
    
    # Retrieval settings
    top_k_retrieval: int = 5
    rerank_top_k: int = 3
    
    # Monitoring
    enable_monitoring: bool = True
    metrics_file: str = "./data/metrics.json"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()

# Ensure data directories exist
for path in [
    settings.chroma_persist_directory,
    settings.pdf_data_path,
    settings.processed_data_path,
    Path(settings.metrics_file).parent
]:
    Path(path).mkdir(parents=True, exist_ok=True)
