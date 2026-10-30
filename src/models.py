"""Validated records shared by ingestion, chunking, retrieval, and RAG."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    document_type: str | None = None
    number_symbol: str | None = None
    title_subject: str | None = None
    issue_date: str | None = None
    issuing_authority: str | None = None
    effective_date: str | None = None
    deadline: str | None = None
    replaces: str | None = None
    amends: str | None = None
    status: str = "chua_xac_minh"
    source_url: str | None = None
    source_path: str
    accessed_at: datetime
    file_hash: str
    page_count: int


class PageRecord(BaseModel):
    document_id: str
    page_no: int = Field(ge=1, description="1-based page index in the original file")
    printed_page: str | None = None
    text: str


class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    page: int
    end_page: int
    section: str | None = None
    strategy: Literal["fixed_size", "recursive"]
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchFilter(BaseModel):
    document_type: str | None = None
    issuing_authority: str | None = None
    topic: str | None = None
    status: str | None = None
    issue_date_from: str | None = None
    issue_date_to: str | None = None
