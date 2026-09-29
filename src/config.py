"""Centralized configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings


ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All configuration loaded from environment variables."""

    # Groq LLM
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    # Pinecone vector database
    pinecone_api_key: str = ""
    pinecone_index_name: str = "opsguard"
    pinecone_namespace: str = "company-documents"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_embedding_model: str = "llama-text-embed-v2"

    # Tavily internet search
    tavily_api_key: str = ""

    # Self-RAG controls
    top_k: int = 5
    max_support_retries: int = 5
    max_retrieval_rewrites: int = 5
    max_web_retries: int = 3

    # Database
    database_path: str = "data/audit.db"

    # LangSmith observability
    langsmith_tracing: bool = False
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: str = ""
    langsmith_project: str = "OpsGuard"

    @property
    def max_web_rewrites(self) -> int:
        """Backward-compatible alias used by graph nodes."""
        return self.max_web_retries

    @property
    def database_file(self) -> Path:
        p = Path(self.database_path)
        return p if p.is_absolute() else ROOT / p

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()