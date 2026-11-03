"""Application configuration loaded from environment variables and .env."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseModel):
    project_root: Path = PROJECT_ROOT
    pdf_dir: Path = PROJECT_ROOT / "data" / "raw"
    web_dir: Path = PROJECT_ROOT / "data" / "raw" / "web"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    evaluation_file: Path = PROJECT_ROOT / "data" / "evaluation" / "questions.csv"
    index_dir: Path = PROJECT_ROOT / "indexes"
    results_dir: Path = PROJECT_ROOT / "results"
    excluded_documents: tuple[str, ...] = ()
    chunk_size: int = Field(default=900, ge=100)
    chunk_overlap: int = Field(default=120, ge=0)
    max_pdf_upload_mb: int = Field(default=50, ge=1)
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_batch_size: int = Field(default=32, ge=1)
    bm25_top_k: int = Field(default=8, ge=1)
    faiss_top_k: int = Field(default=8, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    final_top_k: int = Field(default=5, ge=1)
    token_budget: int = Field(default=3000, ge=100)
    llm_provider: str = "openai_compatible"
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_endpoint: str = "https://api.openai.com/v1/chat/completions"
    llm_reasoning_effort: Literal["minimal", "low", "medium", "high", "none"] | None = None
    llm_timeout: int = Field(default=180, ge=1)

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @classmethod
    def from_env(cls) -> "Settings":
        root = PROJECT_ROOT

        def path_value(key: str, default: Path) -> Path:
            value = os.getenv(key)
            path = Path(value).expanduser() if value else default
            return path if path.is_absolute() else root / path

        excluded = tuple(
            item.strip()
            for item in os.getenv("EXCLUDED_DOCUMENTS", "").split(",")
            if item.strip()
        )
        llm_provider = os.getenv("LLM_PROVIDER", "openai_compatible").strip()
        llm_endpoint = os.getenv("LLM_ENDPOINT")
        if not llm_endpoint:
            llm_base_url = os.getenv("LLM_BASE_URL")
            if llm_base_url:
                llm_endpoint = f"{llm_base_url.rstrip('/')}/chat/completions"
            elif llm_provider.casefold() == "groq":
                llm_endpoint = "https://api.groq.com/openai/v1/chat/completions"
            else:
                llm_endpoint = "https://api.openai.com/v1/chat/completions"
        reasoning_effort = os.getenv("LLM_REASONING_EFFORT")
        if reasoning_effort is None and urlsplit(llm_endpoint).hostname == (
            "generativelanguage.googleapis.com"
        ):
            reasoning_effort = "low"
        return cls(
            project_root=root,
            pdf_dir=path_value("PDF_DIR", root / "data" / "raw"),
            web_dir=path_value("WEB_DIR", root / "data" / "raw" / "web"),
            processed_dir=path_value("PROCESSED_DIR", root / "data" / "processed"),
            evaluation_file=path_value(
                "EVALUATION_FILE", root / "data" / "evaluation" / "questions.csv"
            ),
            index_dir=path_value("INDEX_DIR", root / "indexes"),
            results_dir=path_value("RESULTS_DIR", root / "results"),
            excluded_documents=excluded,
            chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
            max_pdf_upload_mb=int(os.getenv("MAX_PDF_UPLOAD_MB", "50")),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
            bm25_top_k=int(os.getenv("BM25_TOP_K", "8")),
            faiss_top_k=int(os.getenv("FAISS_TOP_K", "8")),
            rrf_k=int(os.getenv("RRF_K", "60")),
            final_top_k=int(os.getenv("FINAL_TOP_K", "5")),
            token_budget=int(os.getenv("TOKEN_BUDGET", "3000")),
            llm_provider=llm_provider,
            llm_api_key=os.getenv("LLM_API_KEY"),
            llm_model=os.getenv("LLM_MODEL"),
            llm_endpoint=llm_endpoint,
            llm_reasoning_effort=reasoning_effort,
            llm_timeout=int(os.getenv("LLM_TIMEOUT", "180")),
        )


settings = Settings.from_env()
