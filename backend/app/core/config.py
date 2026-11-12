import os
from pathlib import Path
from typing import Optional
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Use local cache for huggingface models to prevent slow remote retry timeouts
if "HF_HUB_OFFLINE" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "1"
if "TRANSFORMERS_OFFLINE" not in os.environ:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

class Settings(BaseSettings):
    PROJECT_NAME: str = "Agentic AI System for Autonomous Multi-Task Execution"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = True

    # LLM Settings
    GROQ_API_KEY: str = ""
    DEFAULT_MODEL: str = "qwen/qwen3.8-27b"
    FAST_MODEL: str = "groq/compound-mini"
    MOCK_LLM: bool = False

    # Database
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BASE_DIR}/data/agentic_ai.db"

    # Vector DB & Storage
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "data" / "chroma_db")
    UPLOAD_DIR: str = str(BASE_DIR / "data" / "uploads")
    OUTPUT_DIR: str = str(BASE_DIR / "data" / "outputs")
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    RAG_CHUNK_SIZE: int = 800
    RAG_CHUNK_OVERLAP: int = 150
    RAG_TOP_K: int = 5
    RAG_SIMILARITY_THRESHOLD: float = 0.60

    # Notifications / SMTP
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    EMAILS_FROM_EMAIL: str = "agentic-ai@system.local"
    EMAIL_MOCK_MODE: bool = True

    # Direct aliases for alternative .env naming
    SMTP_SERVER: Optional[str] = None
    SENDER_EMAIL: Optional[str] = None
    SENDER_PASSWORD: Optional[str] = None
    EMAIL_TEST_MODE: Optional[bool] = None

    @model_validator(mode="after")
    def sync_email_aliases(self):
        if self.SMTP_SERVER:
            self.SMTP_HOST = self.SMTP_SERVER
        if self.SENDER_EMAIL:
            self.SMTP_USER = self.SENDER_EMAIL
            self.EMAILS_FROM_EMAIL = self.SENDER_EMAIL
        if self.SENDER_PASSWORD:
            self.SMTP_PASSWORD = self.SENDER_PASSWORD
        if self.EMAIL_TEST_MODE is not None:
            self.EMAIL_MOCK_MODE = bool(self.EMAIL_TEST_MODE)
        return self

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Ensure directories exist
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
