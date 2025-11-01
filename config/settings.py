
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Vector Database
    vector_db_type: str = "qdrant" 


    # Qdrant
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str = ""
    
    
    # Database
    db_type: Literal["postgresql", "mongodb"] = "postgresql"
    database_url: str = "postgresql+asyncpg://ingestion:your_password@postgres:5432/ragdb"
    
    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # App Config
    max_file_size_mb: int = 10
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Redis
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    redis_ttl: int = 3600
    
    # LLM
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = ""
    llm_model: str = "gpt-5-nano"
    llm_temperature: float = 0.7
    max_tokens: int = 1000
    
    # RAG
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.7
    max_context_length: int = 3000
