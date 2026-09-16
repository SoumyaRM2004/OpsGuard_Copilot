from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel
import os
from dotenv import load_dotenv


load_dotenv()


ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseModel):

    # Groq
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    # Pinecone
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "opsguard")
    pinecone_namespace: str = os.getenv(
        "PINECONE_NAMESPACE",
        "company-documents"
    )
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")
    pinecone_embedding_model: str = os.getenv(
        "PINECONE_EMBEDDING_MODEL",
        "llama-text-embed-v2"
    )

    # Internet search
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")

    # Self-RAG controls
    top_k: int = int(os.getenv("TOP_K", "5"))
    max_support_retries: int = int(
        os.getenv("MAX_SUPPORT_RETRIES", "5")
    )
    max_retrieval_rewrites: int = int(
        os.getenv("MAX_RETRIEVAL_REWRITES", "5")
    )
    max_web_rewrites: int = int(
        os.getenv("MAX_WEB_REWRITES", "3")
    )

    @property
    def max_web_retries(self) -> int:
        return self.max_web_rewrites

    # Current database
    database_path: str = os.getenv(
        "DATABASE_PATH",
        "data/audit.db"
    )

    # LangSmith
    langsmith_tracing: bool = os.getenv(
        "LANGSMITH_TRACING",
        "false"
    ).lower() == "true"

    langsmith_endpoint: str = os.getenv(
        "LANGSMITH_ENDPOINT",
        "https://api.smith.langchain.com"
    )

    langsmith_api_key: str = os.getenv(
        "LANGSMITH_API_KEY",
        ""
    )

    langsmith_project: str = os.getenv(
        "LANGSMITH_PROJECT",
        "OpsGuard"
    )

    @property
    def database_file(self) -> Path:
        p = Path(self.database_path)

        return p if p.is_absolute() else ROOT / p


@lru_cache
def get_settings() -> Settings:
    return Settings()