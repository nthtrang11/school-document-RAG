"""Validated records shared by ingestion, chunking, retrieval, and RAG."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_status(cls, value: Any) -> Any:
        if isinstance(value, dict) and "status" in value:
            value = {key: item for key, item in value.items() if key != "status"}
        return value

    document_id: str
    document_type: str | None = None
    number_symbol: str | None = None
    number_symbol_raw: str | None = None
    number_symbol_source: Literal["text_layer", "ocr"] | None = None
    number_symbol_page: int | None = Field(default=None, ge=1)
    title_subject: str | None = None
    issue_date: str | None = None
    issue_date_raw: str | None = None
    issue_year: int | None = None
    issue_date_source: Literal["text_layer", "ocr"] | None = None
    issue_date_page: int | None = Field(default=None, ge=1)
    metadata_verification_status: Literal[
        "unverified", "needs_manual_review", "verified"
    ] = "unverified"
    issuing_authority: str | None = None
    effective_date: str | None = None
    deadline: str | None = None
    replaces: str | None = None
    amends: str | None = None
    source_url: str | None = None
    source_path: str
    accessed_at: datetime
    file_hash: str
    page_count: int


class TableRowRecord(BaseModel):
    table_id: str
    row_index: int = Field(ge=1)
    headers: list[str]
    cells: list[str]
    bboxes: list[tuple[float, float, float, float] | None]

    @model_validator(mode="after")
    def validate_columns(self) -> "TableRowRecord":
        if not (len(self.headers) == len(self.cells) == len(self.bboxes)):
            raise ValueError("Table headers, cells, and bboxes must have matching lengths")
        return self


class PageRecord(BaseModel):
    document_id: str
    page_no: int = Field(ge=1, description="1-based page index in the original file")
    printed_page: str | None = None
    text: str
    table_rows: list[TableRowRecord] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    page: int
    end_page: int
    section: str | None = None
    strategy: Literal["fixed_size", "recursive"]
    search_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchFilter(BaseModel):
    document_type: str | None = None
    issuing_authority: str | None = None
    topic: str | None = None
    issue_date_from: str | None = None
    issue_date_to: str | None = None
