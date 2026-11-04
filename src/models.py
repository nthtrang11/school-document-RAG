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
    number_symbol_source: Literal["text_layer", "ocr", "hybrid"] | None = None
    number_symbol_page: int | None = Field(default=None, ge=1)
    title_subject: str | None = None
    title_subject_raw: str | None = None
    title_subject_corrected: str | None = None
    issue_date: str | None = None
    issue_date_raw: str | None = None
    issue_year: int | None = None
    issue_date_source: Literal["text_layer", "ocr", "hybrid"] | None = None
    issue_date_page: int | None = Field(default=None, ge=1)
    metadata_verification_status: Literal[
        "unverified", "needs_manual_review", "verified", "rejected"
    ] = "unverified"
    metadata_reviewed_by: str | None = None
    metadata_reviewed_at: datetime | None = None
    metadata_review_notes: str | None = None
    issuing_authority: str | None = None
    effective_date: str | None = None
    effective_date_raw: str | None = None
    deadline: str | None = None
    replaces: str | None = None
    amends: str | None = None
    extends: str | None = None
    deadlines: list["DeadlineMetadata"] = Field(default_factory=list)
    document_relations: list["DocumentRelation"] = Field(default_factory=list)
    field_provenance: dict[str, list["MetadataEvidence"]] = Field(default_factory=dict)
    metadata_extraction_warnings: list[str] = Field(default_factory=list)
    source_url: str | None = None
    source_path: str
    accessed_at: datetime
    file_hash: str
    page_count: int


class MetadataEvidence(BaseModel):
    value: str | None = None
    raw_value: str | None = None
    evidence: str
    source_page: int = Field(ge=1)
    source_type: Literal["text_layer", "ocr"] | None = None
    extraction_method: Literal["rule_based", "ocr", "human_review"] = "rule_based"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    bbox: tuple[float, float, float, float] | None = None
    verification_status: Literal["unverified", "verified", "rejected"] = "unverified"


class DeadlineMetadata(BaseModel):
    deadline_type: str | None = None
    value: str | None = None
    raw_value: str
    start_value: str | None = None
    start_raw_value: str | None = None
    source_page: int = Field(ge=1)
    evidence: str
    source_type: Literal["text_layer", "ocr"] | None = None
    extraction_method: Literal["rule_based", "ocr", "human_review"] = "rule_based"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    bbox: tuple[float, float, float, float] | None = None
    verification_status: Literal["unverified", "verified", "rejected"] = "unverified"


class DocumentRelation(BaseModel):
    relation_type: Literal[
        "amends", "replaces", "extends", "revokes", "references", "repeals"
    ]
    referenced_number: str
    raw_value: str
    target_document_id: str | None = None
    target_match_status: Literal["unresolved", "matched", "ambiguous"] = "unresolved"
    target_document_candidates: list[str] = Field(default_factory=list)
    source_page: int = Field(ge=1)
    evidence: str
    source_type: Literal["text_layer", "ocr"] | None = None
    extraction_method: Literal["rule_based", "ocr", "human_review"] = "rule_based"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    bbox: tuple[float, float, float, float] | None = None
    verification_status: Literal["unverified", "verified", "rejected"] = "unverified"


DocumentMetadata.model_rebuild()


class OCRTextRegion(BaseModel):
    text: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    bbox: tuple[float, float, float, float] | None = None


class OCRQuality(BaseModel):
    status: Literal[
        "not_needed",
        "completed",
        "needs_review",
        "failed",
        "unreadable",
        "timeout",
        "retry_pending",
    ]
    recognized_character_count: int = Field(default=0, ge=0)
    valid_character_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    region_count: int = Field(default=0, ge=0)
    reason: str | None = None
    indexable: bool = True


class PageTextQuality(BaseModel):
    text_length: int = Field(default=0, ge=0)
    vn_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    replacement_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    garbage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    word_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    text_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    image_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    layout_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    status: Literal["good", "degraded", "poor", "empty"] = "empty"
    reasons: list[str] = Field(default_factory=list)


class TableRowRecord(BaseModel):
    table_id: str
    table_group_id: str | None = None
    row_index: int = Field(ge=1)
    headers: list[str]
    cells: list[str]
    bboxes: list[tuple[float, float, float, float] | None]
    table_title: str | None = None
    header_rows: list[list[str]] = Field(default_factory=list)
    header_cell_spans: list["TableCellSpan | None"] = Field(default_factory=list)
    cell_states: list[
        Literal[
            "explicit_value",
            "inherited_from_merged_cell",
            "covered_by_merged_cell",
            "empty_or_unknown",
            "text_not_extracted",
        ]
    ] = Field(default_factory=list)
    cell_origins: list[tuple[int, int] | None] = Field(default_factory=list)
    cell_spans: list["TableCellSpan | None"] = Field(default_factory=list)
    table_bbox: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def validate_columns(self) -> "TableRowRecord":
        if not (len(self.headers) == len(self.cells) == len(self.bboxes)):
            raise ValueError("Table headers, cells, and bboxes must have matching lengths")
        for values in (
            self.header_cell_spans,
            self.cell_states,
            self.cell_origins,
            self.cell_spans,
        ):
            if values and len(values) != len(self.cells):
                raise ValueError("Table cell metadata must match the column count")
        return self


class TableCellSpan(BaseModel):
    row_start: int = Field(ge=0)
    row_end: int = Field(ge=1)
    column_start: int = Field(ge=0)
    column_end: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> "TableCellSpan":
        if self.row_end <= self.row_start or self.column_end <= self.column_start:
            raise ValueError("Table cell spans must have positive dimensions")
        return self


class PageRecord(BaseModel):
    document_id: str
    page_no: int = Field(ge=1, description="1-based page index in the original file")
    printed_page: str | None = None
    text: str
    raw_text: str | None = None
    normalized_text: str | None = None
    page_classification: Literal[
        "native_text", "ocr_required", "hybrid", "unreadable"
    ] = "native_text"
    text_source: Literal["text_layer", "ocr", "hybrid"] = "text_layer"
    text_layer_text: str | None = None
    ocr_quality: OCRQuality | None = None
    page_quality: PageTextQuality | None = None
    ocr_regions: list[OCRTextRegion] = Field(default_factory=list)
    page_dimensions: tuple[float, float] | None = None
    page_rotation: int | None = None
    processing_version: str | None = None
    processed_at: datetime | None = None
    ocr_engine: str | None = None
    ocr_dpi: int | None = Field(default=None, ge=1)
    ocr_duration_seconds: float | None = Field(default=None, ge=0.0)
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
