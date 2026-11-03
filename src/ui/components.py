"""Reusable Streamlit components for inline citations and source PDFs."""

from __future__ import annotations

import html
import re
import uuid
from collections.abc import Callable
from typing import Any

import streamlit as st
from markdown_it import MarkdownIt

from src.rag.citation_validator import normalize_citations
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
    .citation-preview__document-title {
        display: block;
        margin: -2px 0 8px;
        color: var(--st-secondary-text-color);
        font-size: 0.8rem;
        line-height: 1.4;
    }
    .citation-preview__page {
        display: block;
        margin-bottom: 12px;
        color: var(--st-secondary-text-color);
        font-size: 0.8rem;
    }
    .citation-preview__columns {
        display: block;
        margin: -4px 0 10px;
        color: var(--st-secondary-text-color);
        font-size: 0.8rem;
        overflow-wrap: anywhere;
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
    .citation-source-link {
        display: inline-flex;
        align-items: center;
        margin-top: 10px;
        padding: 5px 0;
        border: 0;
        color: var(--st-link-color);
        background: transparent;
        font: inherit;
        font-weight: 650;
        cursor: pointer;
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
        preview.style.maxHeight = ""
        const marker = wrapper.querySelector(".citation-marker")
        const markerRect = marker.getBoundingClientRect()
        const previewRect = preview.getBoundingClientRect()
        const margin = 12
        const gap = 8
        const aboveSpace = Math.max(0, markerRect.top - margin - gap)
        const belowSpace = Math.max(
          0,
          window.innerHeight - markerRect.bottom - margin - gap
        )
        const placeBelow = belowSpace >= aboveSpace
        const availableHeight = placeBelow ? belowSpace : aboveSpace
        preview.style.maxHeight = `${Math.min(340, availableHeight)}px`
        const placedPreviewRect = preview.getBoundingClientRect()
        const width = Math.min(previewRect.width, window.innerWidth - margin * 2)
        const left = Math.max(
          margin,
          Math.min(markerRect.left, window.innerWidth - width - margin)
        )
        const top = placeBelow
          ? markerRect.bottom + gap
          : markerRect.top - placedPreviewRect.height - gap
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
        const sourceLink = event.target.closest(".citation-source-link")
        const citation = sourceLink || marker
        if (!citation || !root.contains(citation)) return
        event.preventDefault()
        setTriggerValue("citation", {
          token: citation.dataset.citationToken,
          citation_id: citation.dataset.citationId,
        })
      }
    }
    """,
)

_SOURCE_NAVIGATION = st.components.v2.component(
    "source_document_navigation",
    html='<span class="source-navigation" aria-hidden="true"></span>',
    css=".source-navigation { display: none; }",
    js="""
    export default function (component) {
      const { data, parentElement, setTriggerValue } = component
      const root = parentElement.querySelector(".source-navigation")
      const navigationId = data?.navigationId
      if (!root || !navigationId) return

      const historyMarker = `source-navigation:${navigationId}`
      const currentState = window.history.state
      if (
        currentState?.streamlitSourceNavigation !== historyMarker &&
        root.dataset.navigationId !== navigationId
      ) {
        const nextState =
          currentState && typeof currentState === "object"
            ? { ...currentState }
            : {}
        nextState.streamlitSourceNavigation = historyMarker
        window.history.pushState(nextState, "", window.location.href)
      }
      root.dataset.navigationId = navigationId

      if (root.__sourceNavigationHandler) {
        window.removeEventListener("popstate", root.__sourceNavigationHandler)
      }
      root.__sourceNavigationHandler = () => {
        if (root.dataset.backTriggered === navigationId) return
        root.dataset.backTriggered = navigationId
        window.removeEventListener("popstate", root.__sourceNavigationHandler)
        root.__sourceNavigationHandler = null
        setTriggerValue("back", navigationId)
      }
      window.addEventListener("popstate", root.__sourceNavigationHandler)

      if (
        data?.backRequested &&
        root.dataset.backRequestHandled !== navigationId
      ) {
        root.dataset.backRequestHandled = navigationId
        if (
          window.history.state?.streamlitSourceNavigation === historyMarker
        ) {
          window.history.back()
        } else {
          root.__sourceNavigationHandler()
        }
      }
    }
    """,
)

_SEARCH_INPUT_FOCUS = st.components.v2.component(
    "search_input_enter_blur",
    html='<span class="search-input-focus" aria-hidden="true"></span>',
    css=".search-input-focus { display: none; }",
    js="""
    export default function () {
      if (window.__searchInputEnterBlurHandler) return
      window.__searchInputEnterBlurHandler = (event) => {
        const input = event.target
        if (
          event.key !== "Enter" ||
          event.isComposing ||
          event.shiftKey ||
          !(input instanceof HTMLInputElement)
        ) return
        const label =
          input.getAttribute("aria-label") ||
          input.closest('[data-testid="stTextInput"]')?.querySelector("label")?.textContent?.trim()
        if (label !== "Tìm kiếm thông tin") return
        window.setTimeout(() => {
          if (document.activeElement === input) input.blur()
        }, 0)
      }
      document.addEventListener(
        "keydown",
        window.__searchInputEnterBlurHandler,
        true
      )
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

    renderable_ids = {
        citation_id
        for citation_id, source in source_by_id.items()
        if source.get("document_id")
        and source.get("chunk_id")
        and isinstance(source.get("evidence_text"), str)
        and source["evidence_text"]
        and isinstance(source.get("page"), int)
        and source["page"] >= 1
        and (
            source.get("page_count") is None
            or (
                isinstance(source.get("page_count"), int)
                and source["page"] <= source["page_count"]
            )
        )
        and (source.get("source_path") or source.get("source_url"))
    }
    answer, _ = normalize_citations(answer, renderable_ids)

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

        document_title = str(
            source.get("document_title") or source.get("title") or "Tài liệu nguồn"
        )
        source_filename = str(
            source.get("source_filename")
            or source.get("source_path")
            or document_title
        ).replace("\\", "/").rsplit("/", 1)[-1]
        record = {
            **source,
            "citation_id": citation_id,
            "document_title": document_title,
            "pdf_path": source.get("pdf_path") or source.get("source_path"),
            "source": source.get("source")
            or source.get("pdf_path")
            or source.get("source_path")
            or source.get("source_url"),
            "bbox": source.get("bbox"),
        }
        placeholder = f"CITATIONPLACEHOLDER{len(replacements)}END"
        preview_id = f"citation-preview-{len(replacements)}"
        escaped_title = html.escape(source_filename, quote=True)
        escaped_document_title = html.escape(document_title, quote=True)
        escaped_evidence = html.escape(evidence, quote=True)
        escaped_token = html.escape(token, quote=True)
        page_text = f"Trang {page:02d}"
        headers = source.get("table_headers")
        columns_html = ""
        if isinstance(headers, list) and all(
            isinstance(header, str) for header in headers
        ):
            labels = " · ".join(header for header in headers if header)
            if labels:
                columns_html = (
                    '<span class="citation-preview__columns">'
                    f"Cột theo thứ tự: {html.escape(labels, quote=True)}"
                    "</span>"
                )
        replacement = (
            '<span class="citation-wrap">'
            f'<a class="citation-marker" href="#" data-citation-token="{escaped_token}" '
            f'data-citation-id="{citation_id}" aria-describedby="{preview_id}">'
            f"[{citation_id}]</a>"
            f'<button class="citation-preview-toggle" type="button" '
            f'aria-label="Xem trước {escaped_title}, {page_text}" '
            f'aria-controls="{preview_id}" aria-expanded="false">i</button>'
            f'<span class="citation-preview" id="{preview_id}" role="dialog" '
            f'aria-label="Bằng chứng từ {escaped_title}">'
            f'<strong class="citation-preview__title">{escaped_title}</strong>'
            f'<span class="citation-preview__document-title">{escaped_document_title}</span>'
            f'<span class="citation-preview__page">{page_text}</span>'
            '<span class="citation-preview__label">Nội dung trích dẫn</span>'
            f"{columns_html}"
            f'<span class="citation-preview__quote">{escaped_evidence}</span>'
            f'<button class="citation-source-link" type="button" '
            f'data-citation-token="{escaped_token}" data-citation-id="{citation_id}">'
            "Xem nguồn</button>"
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
        "navigation_id": uuid.uuid4().hex,
    }


def _return_to_search() -> None:
    st.session_state.pop("active_citation", None)
    st.session_state.pop("active_citation_error", None)
    st.session_state.pop("source_back_requested", None)


def _handle_source_navigation(component_key: str, navigation_id: str) -> None:
    component_state = st.session_state.get(component_key, {})
    payload = component_state.get("back") if isinstance(component_state, dict) else None
    active_citation = st.session_state.get("active_citation", {})
    if (
        payload == navigation_id
        and isinstance(active_citation, dict)
        and active_citation.get("navigation_id") == navigation_id
    ):
        _return_to_search()


def render_source_navigation(
    navigation_id: str, *, back_requested: bool = False
) -> None:
    component_key = f"source-navigation-{navigation_id}"
    _SOURCE_NAVIGATION(
        key=component_key,
        data={
            "navigationId": navigation_id,
            "backRequested": back_requested,
        },
        on_back_change=lambda: _handle_source_navigation(
            component_key, navigation_id
        ),
    )


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


def render_search_input_focus_behavior() -> None:
    _SEARCH_INPUT_FOCUS(key="search-input-enter-blur", data={})


def render_pdf_source(
    source: dict[str, Any], on_back: Callable[[], None] | None = None
) -> None:
    source_path = source.get("pdf_path") or source.get("source_path")
    page = source.get("page")
    if not source_path or not isinstance(page, int) or page < 1:
        st.error("Citation này chưa có đường dẫn PDF nguồn hoặc số trang hợp lệ.")
        if on_back:
            st.button("Quay lại tra cứu", on_click=on_back)
        return

    try:
        pdf_path = resolve_pdf_path(str(source_path))
        pdf_bytes, bbox, highlight_status = build_pdf_view(
            pdf_path,
            page,
            str(source.get("evidence_text") or ""),
            source.get("table_locations"),
            source.get("evidence_segments"),
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error(f"Không thể mở tài liệu PDF gốc: {exc}")
        if on_back:
            st.button("Quay lại tra cứu", on_click=on_back)
        return

    if highlight_status == "partial_match":
        st.caption(
            "Chỉ khớp một phần bằng chứng; hệ thống không đánh dấu để tránh highlight sai."
        )
    elif bbox is None:
        st.caption(
            "Đã mở đúng trang; chưa thể xác định chính xác đoạn bằng chứng để đánh dấu."
        )
    else:
        st.caption("Đoạn bằng chứng được đánh dấu trong PDF.")
    if on_back:
        st.button("Quay lại tra cứu", on_click=on_back)
    st.pdf(pdf_bytes, height=800, alt="PDF nguồn của citation")
