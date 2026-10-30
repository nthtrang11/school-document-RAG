"""Reusable Streamlit components for inline citations and source PDFs."""

from __future__ import annotations

import html
import re
from typing import Any

import streamlit as st
from markdown_it import MarkdownIt

from src.rag.source_viewer import build_pdf_view, resolve_pdf_path

_CITATION = re.compile(r"\[(E\d+)\]")
_CITATION_ID = re.compile(r"^\[?(E\d+)\]?$")
_MARKDOWN = MarkdownIt("default", {"html": False})

_INLINE_ANSWER = st.components.v2.component(
    "inline_citation_answer",
    html='<div class="citation-answer"></div>',
    css="""
    .citation-answer {
        color: var(--st-text-color);
        font-family: var(--st-font);
        font-size: 1rem;
        line-height: 1.65;
        overflow: visible;
    }
    .citation-answer p { margin: 0 0 0.75rem; }
    .citation-answer p:last-child { margin-bottom: 0; }
    .citation-answer a.citation-marker {
        color: var(--st-link-color);
        font-weight: 650;
        text-decoration: underline;
        text-underline-offset: 0.14em;
        cursor: pointer;
    }
    .citation-wrap {
        display: inline;
        position: relative;
        white-space: normal;
    }
    .citation-preview {
        display: none;
        position: fixed;
        box-sizing: border-box;
        z-index: 2147483000;
        width: min(360px, calc(100vw - 24px));
        max-height: min(340px, calc(100vh - 24px));
        overflow-y: auto;
        padding: 14px 16px;
        color: var(--st-text-color);
        background: var(--st-secondary-background-color);
        border: 1px solid var(--st-border-color);
        border-radius: 12px;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.2);
        text-align: left;
        white-space: normal;
        font-weight: 400;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .citation-preview.is-open { display: block; }
    .citation-preview__title {
        display: block;
        margin-bottom: 2px;
        color: var(--st-heading-color);
        font-size: 0.98rem;
        font-weight: 700;
        line-height: 1.35;
        overflow-wrap: anywhere;
    }
    .citation-preview__page {
        display: block;
        margin-bottom: 12px;
        color: var(--st-secondary-text-color);
        font-size: 0.8rem;
    }
    .citation-preview__label {
        display: block;
        margin-bottom: 5px;
        color: var(--st-secondary-text-color);
        font-size: 0.75rem;
        font-weight: 650;
        letter-spacing: 0.025em;
        text-transform: uppercase;
    }
    .citation-preview__quote {
        display: block;
        max-height: 170px;
        overflow-y: auto;
        padding: 10px 12px;
        border-left: 3px solid var(--st-primary-color);
        border-radius: 0 7px 7px 0;
        background: var(--st-background-color);
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        font-weight: 400;
        line-height: 1.6;
        user-select: text;
    }
    .citation-preview-toggle {
        display: inline-flex;
        width: 1.25rem;
        height: 1.25rem;
        align-items: center;
        justify-content: center;
        margin-left: 1px;
        padding: 0;
        border: 0;
        border-radius: 50%;
        color: var(--st-secondary-text-color);
        background: transparent;
        font: inherit;
        font-size: 0.75rem;
        line-height: 1;
        vertical-align: text-top;
        cursor: pointer;
        opacity: 0;
    }
    .citation-wrap:hover .citation-preview-toggle,
    .citation-preview-toggle:focus-visible { opacity: 1; }
    .citation-preview-toggle:focus-visible {
        outline: 2px solid var(--st-primary-color);
        outline-offset: 1px;
    }
    @media (hover: none) {
        .citation-preview-toggle { opacity: 1; }
    }
    """,
    js="""
    export default function (component) {
      const { data, parentElement, setTriggerValue } = component
      const root = parentElement.querySelector(".citation-answer")
      if (!root) return

      const nextHtml = data?.html ?? ""
      if (root.dataset.renderedHtml !== nextHtml) {
        root.innerHTML = nextHtml
        root.dataset.renderedHtml = nextHtml
      }

      const closePreview = (wrapper) => {
        const preview = wrapper?.querySelector(".citation-preview")
        if (wrapper?.dataset.pinned === "true") return
        if (preview) preview.classList.remove("is-open")
        const toggle = wrapper?.querySelector(".citation-preview-toggle")
        if (toggle) toggle.setAttribute("aria-expanded", "false")
      }

      const forceClosePreview = (wrapper) => {
        if (wrapper) delete wrapper.dataset.pinned
        const preview = wrapper?.querySelector(".citation-preview")
        if (preview) preview.classList.remove("is-open")
        const toggle = wrapper?.querySelector(".citation-preview-toggle")
        if (toggle) toggle.setAttribute("aria-expanded", "false")
      }

      const placePreview = (wrapper) => {
        const preview = wrapper.querySelector(".citation-preview")
        if (!preview) return
        preview.classList.add("is-open")
        preview.style.visibility = "hidden"
        const marker = wrapper.querySelector(".citation-marker")
        const markerRect = marker.getBoundingClientRect()
        const previewRect = preview.getBoundingClientRect()
        const margin = 12
        const width = Math.min(previewRect.width, window.innerWidth - margin * 2)
        const height = Math.min(previewRect.height, window.innerHeight - margin * 2)
        const left = Math.max(
          margin,
          Math.min(markerRect.left, window.innerWidth - width - margin)
        )
        let top = markerRect.bottom + 8
        if (top + height > window.innerHeight - margin) {
          top = Math.max(margin, markerRect.top - height - 8)
        }
        preview.style.left = `${left}px`
        preview.style.top = `${top}px`
        preview.style.visibility = "visible"
      }

      root.onpointerover = (event) => {
        if (window.matchMedia("(hover: none)").matches) return
        const wrapper = event.target.closest(".citation-wrap")
        if (!wrapper || !root.contains(wrapper) || wrapper.contains(event.relatedTarget)) return
        placePreview(wrapper)
      }

      root.onpointerout = (event) => {
        if (window.matchMedia("(hover: none)").matches) return
        const wrapper = event.target.closest(".citation-wrap")
        if (!wrapper || wrapper.contains(event.relatedTarget)) return
        closePreview(wrapper)
      }

      root.onfocusin = (event) => {
        if (window.matchMedia("(hover: none)").matches) return
        const wrapper = event.target.closest(".citation-wrap")
        if (wrapper) placePreview(wrapper)
      }

      root.onfocusout = (event) => {
        if (window.matchMedia("(hover: none)").matches) return
        const wrapper = event.target.closest(".citation-wrap")
        if (wrapper && !wrapper.contains(event.relatedTarget)) closePreview(wrapper)
      }

      root.onclick = (event) => {
        const toggle = event.target.closest(".citation-preview-toggle")
        if (toggle) {
          event.preventDefault()
          event.stopPropagation()
          const wrapper = toggle.closest(".citation-wrap")
          const preview = wrapper?.querySelector(".citation-preview")
          if (wrapper?.dataset.pinned === "true") {
            forceClosePreview(wrapper)
          } else if (wrapper) {
            wrapper.dataset.pinned = "true"
            placePreview(wrapper)
            toggle.setAttribute("aria-expanded", "true")
          }
          return
        }

        const marker = event.target.closest(".citation-marker")
        if (!marker || !root.contains(marker)) return
        event.preventDefault()
        setTriggerValue("citation", {
          token: marker.dataset.citationToken,
          citation_id: marker.dataset.citationId,
        })
      }
    }
    """,
)


def _normalize_citation_id(citation_id: Any) -> str | None:
    if not isinstance(citation_id, str):
        return None
    match = _CITATION_ID.fullmatch(citation_id.strip())
    return match.group(1) if match else None


def build_inline_answer_html(
    answer: str, sources: list[dict[str, Any]], token: str
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Render Markdown safely while replacing mapped citation markers with UI elements."""
    source_by_id: dict[str, dict[str, Any]] = {}
    for source in sources:
        citation_id = _normalize_citation_id(
            source.get("citation_id") or source.get("evidence_id")
        )
        if citation_id:
            source_by_id[citation_id] = source

    linked_sources: dict[str, dict[str, Any]] = {}
    replacements: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        citation_id = match.group(1)
        source = source_by_id.get(citation_id)
        page = source.get("page") if source else None
        page_count = source.get("page_count") if source else None
        evidence = source.get("evidence_text") if source else None
        if not source or not (
            source.get("document_id")
            and source.get("chunk_id")
            and isinstance(evidence, str)
            and evidence
            and isinstance(page, int)
            and page >= 1
            and (
                page_count is None
                or (isinstance(page_count, int) and page <= page_count)
            )
        ):
            return match.group(0)

        title = str(
            source.get("document_title") or source.get("title") or "Tài liệu nguồn"
        )
        record = {
            **source,
            "citation_id": citation_id,
            "document_title": title,
            "pdf_path": source.get("pdf_path") or source.get("source_path"),
            "source": source.get("source")
            or source.get("pdf_path")
            or source.get("source_path")
            or source.get("source_url"),
            "bbox": source.get("bbox"),
        }
        placeholder = f"CITATIONPLACEHOLDER{len(replacements)}END"
        preview_id = f"citation-preview-{len(replacements)}"
        escaped_title = html.escape(title, quote=True)
        escaped_evidence = html.escape(evidence, quote=True)
        escaped_token = html.escape(token, quote=True)
        page_text = f"Trang {page:02d}"
        replacement = (
            '<span class="citation-wrap">'
            f'<a class="citation-marker" href="#" data-citation-token="{escaped_token}" '
            f'data-citation-id="{citation_id}" aria-describedby="{preview_id}">'
            f"[{citation_id}]</a>"
            f'<button class="citation-preview-toggle" type="button" '
            f'aria-label="Xem trước {escaped_title}, {page_text}" '
            f'aria-controls="{preview_id}" aria-expanded="false">i</button>'
            f'<span class="citation-preview" id="{preview_id}" role="tooltip">'
            f'<strong class="citation-preview__title">{escaped_title}</strong>'
            f'<span class="citation-preview__page">{page_text}</span>'
            '<span class="citation-preview__label">Nội dung trích dẫn</span>'
            f'<span class="citation-preview__quote">{escaped_evidence}</span>'
            "</span></span>"
        )
        linked_sources[citation_id] = record
        replacements[placeholder] = replacement
        return placeholder

    markdown = _CITATION.sub(replace, answer)
    rendered = _MARKDOWN.render(markdown)
    for placeholder, replacement in replacements.items():
        rendered = rendered.replace(placeholder, replacement)
    return rendered, linked_sources


def register_citations(
    registry: dict[str, dict[str, dict[str, Any]]],
    token: str,
    sources: dict[str, dict[str, Any]],
) -> None:
    """Register records under a stable answer token without replacing other answers."""
    registry.setdefault(token, {}).update(sources)


def lookup_citation(
    registry: dict[str, dict[str, dict[str, Any]]], token: Any, citation_id: Any
) -> dict[str, Any] | None:
    normalized_id = _normalize_citation_id(citation_id)
    if not isinstance(token, str) or not normalized_id:
        return None
    return registry.get(token, {}).get(normalized_id)


def _handle_citation_click(component_key: str) -> None:
    component_state = st.session_state.get(component_key, {})
    payload = component_state.get("citation") if isinstance(component_state, dict) else None
    if not isinstance(payload, dict):
        return

    registry = st.session_state.get("citation_registry", {})
    record = lookup_citation(
        registry, payload.get("token"), payload.get("citation_id")
    )
    if record is None:
        st.session_state["active_citation_error"] = (
            "Không tìm thấy citation mapping hợp lệ trong câu trả lời hiện tại."
        )
        st.session_state.pop("active_citation", None)
        return

    st.session_state.pop("active_citation_error", None)
    st.session_state["active_citation"] = {
        "token": payload["token"],
        "citation_id": record["citation_id"],
    }


def render_inline_answer(
    answer: str, sources: list[dict[str, Any]], token: str
) -> None:
    rendered_html, linked_sources = build_inline_answer_html(answer, sources, token)
    registry = st.session_state.setdefault("citation_registry", {})
    register_citations(registry, token, linked_sources)

    component_key = f"inline-answer-{token}"
    _INLINE_ANSWER(
        key=component_key,
        data={"html": rendered_html},
        on_citation_change=lambda: _handle_citation_click(component_key),
    )


def render_pdf_source(source: dict[str, Any]) -> None:
    source_path = source.get("pdf_path") or source.get("source_path")
    page = source.get("page")
    if not source_path or not isinstance(page, int) or page < 1:
        st.error("Citation này chưa có đường dẫn PDF nguồn hoặc số trang hợp lệ.")
        return

    try:
        pdf_path = resolve_pdf_path(str(source_path))
        pdf_bytes, bbox = build_pdf_view(
            pdf_path,
            page,
            str(source.get("evidence_text") or ""),
            source.get("bbox"),
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error(f"Không thể mở tài liệu PDF gốc: {exc}")
        return

    if bbox is None:
        st.caption("Đã mở đúng trang; không tìm thấy đoạn khớp để đánh dấu.")
    else:
        st.caption("Đoạn bằng chứng được đánh dấu trong PDF.")
    st.pdf(pdf_bytes, height=800)
