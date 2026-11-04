"""Vietnamese school-document search and grounded Q&A portal."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.config import settings
from src.indexing.index_manager import IndexBundle, load_strategy
from src.ingestion.upload_service import (
    UploadPayload,
    load_upload_registry,
    process_pdf_uploads,
)
from src.models import DocumentMetadata
from src.ingestion.metadata_review import (
    load_metadata_records,
    save_metadata_review_and_reindex,
)
from src.rag.answer_generator import generate_answer
from src.rag.citation_validator import retrieval_only_response
from src.rag.context_builder import build_context
from src.retrieval.hybrid_retriever import HybridRetriever
from src.ui.components import (
    lookup_citation,
    render_inline_answer,
    render_pdf_source,
    render_search_input_focus_behavior,
    render_source_navigation,
)

st.set_page_config(page_title="Cổng tra cứu văn bản", page_icon=":material/school:", layout="wide")

citation_registry = st.session_state.setdefault("citation_registry", {})
active_citation = st.session_state.get("active_citation")


def _request_source_back() -> None:
    active = st.session_state.get("active_citation", {})
    navigation_id = active.get("navigation_id") if isinstance(active, dict) else None
    if isinstance(navigation_id, str):
        st.session_state["source_back_requested"] = navigation_id


if active_citation:
    navigation_id = active_citation.get("navigation_id")
    if not isinstance(navigation_id, str) or not navigation_id:
        navigation_id = uuid.uuid4().hex
        active_citation["navigation_id"] = navigation_id
    st.html(
        """
        <style>
        [data-testid="stMainBlockContainer"]:has(.st-key-source-document-view) {
            padding-top: 1rem !important;
        }
        .st-key-source-document-view {
            padding-top: 0;
        }
        </style>
        """
    )
    with st.container(key="source-document-view", gap="small"):
        st.title("Tài liệu nguồn")
        render_source_navigation(
            navigation_id,
            back_requested=st.session_state.get("source_back_requested")
            == navigation_id,
        )
        source = lookup_citation(
            citation_registry,
            active_citation.get("token"),
            active_citation.get("citation_id"),
        )
        if source:
            title = source.get("document_title") or "Tài liệu nguồn"
            st.subheader(title)
            st.caption(
                f"Trang {source['page']} · Citation [{source['citation_id']}]"
            )
            render_pdf_source(
                source,
                on_back=_request_source_back,
            )
        else:
            st.error("Không tìm thấy citation mapping hợp lệ trong phiên hiện tại.")
            st.button(
                "Quay lại tra cứu",
                key="source-back-to-search",
                on_click=_request_source_back,
            )
    st.stop()
if citation_error := st.session_state.pop("active_citation_error", None):
    st.error(citation_error)


@st.cache_resource
def get_index(strategy: str) -> IndexBundle:
    return load_strategy(strategy)


@st.cache_resource
def get_retriever(strategy: str) -> HybridRetriever:
    return HybridRetriever(get_index(strategy))


def _document_catalog(bundle: IndexBundle) -> pd.DataFrame:
    uploaded_records = load_upload_registry()
    completed_uploads = {
        str(record.get("document_id")): record
        for record in uploaded_records
        if record.get("document_id")
        and record.get("processing_status") == "completed"
    }
    rows = {}
    for chunk in bundle.chunks:
        meta = chunk.metadata
        upload = completed_uploads.get(chunk.document_id, {})
        source_path = meta.get("source_path") or ""
        rows[chunk.document_id] = {
            "document_id": chunk.document_id,
            "filename": upload.get("filename") or Path(source_path).name or "Chưa xác định",
            "title": meta.get("title_subject") or "Chưa xác định tiêu đề",
            "number_symbol": meta.get("number_symbol") or "Chưa xác định",
            "issue_date": meta.get("issue_date") or "Chưa xác định",
            "document_type": meta.get("document_type") or "Chưa xác định",
            "issuing_authority": meta.get("issuing_authority") or "Chưa xác định",
            "page_count": meta.get("page_count") or "Chưa xác định",
            "uploaded_at": upload.get("uploaded_at") or "Chưa xác định",
            "processing_status": "completed",
            "source_path": source_path,
            "source_url": meta.get("source_url") or "",
        }
    return pd.DataFrame(rows.values())


st.title("Cổng tra cứu thông tin, thông báo và văn bản")
st.caption("Tra cứu trên tài liệu đã nạp; câu trả lời cần có nguồn để kiểm chứng.")

with st.popover("+ Tải tài liệu mới"):
    st.write("Chọn một hoặc nhiều PDF. Tài liệu chỉ được công bố sau khi cả hai chỉ mục hoàn tất.")
    upload_version = st.session_state.setdefault("upload_widget_version", 0)
    uploaded_files = st.file_uploader(
        "Tệp PDF",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_upload_{upload_version}",
        max_upload_size=settings.max_pdf_upload_mb,
        help=f"Tối đa {settings.max_pdf_upload_mb} MB cho mỗi tệp.",
    )
    upload_columns = st.columns(2)
    confirm_upload = upload_columns[0].button(
        "Tải lên và xử lý",
        type="primary",
        disabled=not uploaded_files,
        key="confirm_pdf_upload",
    )
    cancel_upload = upload_columns[1].button("Hủy", key="cancel_pdf_upload")

    if cancel_upload:
        st.session_state["upload_widget_version"] = upload_version + 1
        st.rerun()

    if confirm_upload and uploaded_files:
        with st.status("Đang kiểm tra tài liệu...", expanded=True) as upload_status:
            results = process_pdf_uploads(
                [
                    UploadPayload(file.name, file.getvalue())
                    for file in uploaded_files
                ],
                on_stage=lambda _stage, detail: upload_status.update(label=detail),
            )
            final_state = (
                "complete"
                if all(result.status in {"completed", "duplicate"} for result in results)
                else "error"
            )
            upload_status.update(
                label=(
                    "Đã xử lý xong."
                    if final_state == "complete"
                    else "Có tài liệu chưa được tích hợp."
                ),
                state=final_state,
                expanded=False,
            )
        st.session_state["upload_results"] = [
            {
                "filename": result.filename,
                "status": result.status,
                "message": result.message,
                "document_id": result.document_id,
                "page_count": result.page_count,
                "ocr_pages_needing_review": result.ocr_pages_needing_review,
            }
            for result in results
        ]
        st.session_state["upload_widget_version"] = upload_version + 1
        if any(result.status == "completed" for result in results):
            get_index.clear()
            get_retriever.clear()
        st.rerun()

if st.session_state.get("upload_results"):
    for upload_result in st.session_state.pop("upload_results"):
        detail = upload_result["message"]
        if upload_result.get("document_id"):
            detail += f" · Document ID `{upload_result['document_id']}`"
        if upload_result.get("page_count"):
            detail += f" · {upload_result['page_count']} trang"
        if upload_result.get("ocr_pages_needing_review"):
            detail += (
                " · OCR cần rà soát trang "
                + ", ".join(
                    str(page)
                    for page in upload_result["ocr_pages_needing_review"]
                )
            )
        if upload_result["status"] == "completed" and not upload_result.get(
            "ocr_pages_needing_review"
        ):
            st.success(f"{upload_result['filename']}: {detail}")
        elif upload_result["status"] == "completed":
            st.warning(f"{upload_result['filename']}: {detail}")
        elif upload_result["status"] == "duplicate":
            st.warning(f"{upload_result['filename']}: {detail}")
        else:
            st.error(f"{upload_result['filename']}: {detail}")

available = [
    strategy
    for strategy in ("fixed_size", "recursive")
    if (settings.index_dir / strategy / "manifest.json").exists()
]

with st.sidebar:
    st.header("Bộ lọc")
    strategy = st.selectbox(
        "Chiến lược chunking",
        options=["fixed_size", "recursive"],
        format_func=lambda value: "Fixed-size" if value == "fixed_size" else "Recursive",
    )
    st.caption("Lọc metadata chỉ áp dụng các trường đã trích xuất được.")

if not available:
    st.info("Chưa có index. Hãy nạp PDF rồi chạy ingestion và xây dựng index theo README.")
    st.code("python -m src.ingestion.ingestion_pipeline\npython -m src.indexing.index_manager --strategy both")
    st.stop()
if strategy not in available:
    st.warning(f"Chưa có index {strategy}. Chọn chiến lược đã được lập chỉ mục.")
    strategy = available[0]

try:
    bundle = get_index(strategy)
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Không thể tải index: {exc}")
    st.stop()

catalog = _document_catalog(bundle)
with st.sidebar:
    type_options = sorted(value for value in catalog["document_type"].unique() if value != "Chưa xác định")
    authority_options = sorted(
        value for value in catalog["issuing_authority"].unique() if value != "Chưa xác định"
    )
    selected_type = st.selectbox("Loại văn bản", ["Tất cả", *type_options])
    selected_authority = st.selectbox("Đơn vị ban hành", ["Tất cả", *authority_options])
    date_range = st.date_input("Khoảng ngày ban hành", value=(), format="DD/MM/YYYY")
    topic = st.text_input("Chủ đề", placeholder="Từ khóa về nội dung")
    filters = {
        "document_type": None if selected_type == "Tất cả" else selected_type,
        "issuing_authority": None if selected_authority == "Tất cả" else selected_authority,
        "topic": topic or None,
    }
    if isinstance(date_range, tuple) and len(date_range) == 2:
        filters["issue_date_from"] = date_range[0].isoformat()
        filters["issue_date_to"] = date_range[1].isoformat()

metric_cols = st.columns(4)
metric_cols[0].metric("Tài liệu", len(catalog))
metric_cols[1].metric("Trang", len({
    (chunk.document_id, chunk.page) for chunk in bundle.chunks
}))
metric_cols[2].metric("Chunks", len(bundle.chunks))
summary_path = settings.processed_dir / "ingestion_summary.json"
rejected_count = 0
if summary_path.exists():
    ingestion_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rejected_count = ingestion_summary.get("rejected_documents", 0)
metric_cols[3].metric("Tài liệu bị loại", rejected_count)
st.caption(f"Index `{strategy}` · embedding `{bundle.model_name}`")

with st.form("search_form"):
    if "search_question_widget" not in st.session_state:
        previous_result = st.session_state.get("last_search_result")
        st.session_state["search_question_widget"] = (
            previous_result.get("query", "")
            if isinstance(previous_result, dict)
            else ""
        )
    question = st.text_input(
        "Tìm kiếm thông tin",
        placeholder="Ví dụ: Thời hạn đăng ký học kỳ I năm học 2026–2027?",
        key="search_question_widget",
    )
    submitted = st.form_submit_button("Tra cứu", type="primary")
render_search_input_focus_behavior()

if not question:
    st.subheader("Gợi ý câu hỏi")
    st.markdown(
        "- Lịch đăng ký học kỳ I năm học 2026–2027 có những mốc nào?\n"
        "- Thông báo về gia hạn bảo hiểm y tế năm 2027 đề cập đến nội dung gì?\n"
        "- Vì sao nhà trường lấy ý kiến phản hồi về hoạt động giảng dạy lần 2?"
    )

def answer_query(query: str) -> tuple[dict, list[dict]]:
    retrieval_result = get_retriever(strategy).search(query, filters=filters)
    evidence_result = build_context(retrieval_result["results"], query=query)
    try:
        response = generate_answer(query, evidence_result)
    except (RuntimeError, ValueError, OSError) as exc:
        response = retrieval_only_response(evidence_result)
        response["error"] = str(exc)
    return response, evidence_result


def render_query_result(query: str) -> None:
    try:
        response, _ = answer_query(query)
        st.session_state["last_search_result"] = {
            "query": query,
            "response": response,
            "citation_token": uuid.uuid4().hex,
        }
    except (RuntimeError, ValueError, OSError) as exc:
        st.error(f"Không thể hoàn tất tra cứu: {exc}")

if submitted and question.strip():
    render_query_result(question.strip())

search_result = st.session_state.get("last_search_result")
if search_result:
    response = search_result["response"]
    st.subheader("Câu trả lời")
    if response.get("error"):
        st.error(response["error"])
    render_inline_answer(
        response["answer"], response.get("sources", []), search_result["citation_token"]
    )
    if not response["generated"]:
        st.info("Câu trả lời tự động chưa khả dụng; kết quả truy xuất chưa được diễn giải.")
    if response["invalid_citations"]:
        st.warning("Một số citation không có mapping trong evidence nên được giữ dạng văn bản.")

st.subheader("Chat tra cứu")
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            message.setdefault("citation_token", uuid.uuid4().hex)
            render_inline_answer(
                message["content"],
                message.get("sources", []),
                message["citation_token"],
            )
        else:
            st.write(message["content"])

chat_query = st.chat_input("Đặt câu hỏi về thông báo, học phí, lịch học...")
if chat_query:
    st.session_state.messages.append({"role": "user", "content": chat_query})
    with st.chat_message("user"):
        st.write(chat_query)
    with st.chat_message("assistant"):
        try:
            response, _ = answer_query(chat_query)
            if response.get("error"):
                st.error(response["error"])
            sources = response.get("sources", [])
            message_id = len(st.session_state.messages)
            token = uuid.uuid4().hex
            render_inline_answer(response["answer"], sources, token)
            if not response["generated"]:
                st.info("LLM chưa cấu hình; đây là kết quả truy xuất, chưa phải câu trả lời sinh.")
            if response["invalid_citations"]:
                st.warning("Một số citation không có mapping trong evidence nên được giữ dạng văn bản.")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": sources,
                    "message_id": message_id,
                    "citation_token": token,
                }
            )
        except (RuntimeError, ValueError, OSError) as exc:
            st.error(f"Không thể hoàn tất tra cứu: {exc}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Không thể tra cứu: {exc}", "sources": []}
            )

st.subheader("Danh mục tài liệu")
st.dataframe(
    catalog[
        [
            "filename",
            "title",
            "document_type",
            "number_symbol",
            "issue_date",
            "issuing_authority",
            "page_count",
            "uploaded_at",
            "processing_status",
        ]
    ],
    hide_index=True,
)

metadata_path = settings.processed_dir / "documents.jsonl"
metadata_records = load_metadata_records(metadata_path)
if metadata_records:
    with st.expander("Xác minh metadata", expanded=False):
        labels = {
            record.document_id: (
                f"{record.document_id} · "
                f"{record.title_subject or record.number_symbol or 'Chưa có tiêu đề'} "
                f"({record.metadata_verification_status})"
            )
            for record in metadata_records
        }
        selected_document_id = st.selectbox(
            "Tài liệu cần rà soát",
            list(labels),
            format_func=lambda value: labels[value],
            key="metadata_review_document",
        )
        selected_metadata = next(
            record
            for record in metadata_records
            if record.document_id == selected_document_id
        )
        if selected_metadata.field_provenance:
            st.json({
                field: [
                    evidence.model_dump(mode="json")
                    for evidence in entries
                ]
                for field, entries in selected_metadata.field_provenance.items()
            })
        else:
            st.info("Chưa có provenance; cần bổ sung bằng chứng trước khi xác minh.")

        with st.form(f"metadata_review_form_{selected_document_id}"):
            reviewed_json = st.text_area(
                "Metadata JSON để xác nhận hoặc chỉnh sửa",
                value=json.dumps(
                    selected_metadata.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                ),
                height=320,
                key=f"metadata_review_json_{selected_document_id}",
                help=(
                    "Khi sửa giá trị, cập nhật field_provenance tương ứng và giữ "
                    "bằng chứng/trang nguồn."
                ),
            )
            reviewer = st.text_input("Người rà soát")
            review_notes = st.text_input("Ghi chú rà soát")
            review_status = st.selectbox(
                "Kết luận",
                ["verified", "rejected"],
                format_func=lambda value: (
                    "Đã xác minh" if value == "verified" else "Từ chối giá trị"
                ),
            )
            review_submitted = st.form_submit_button(
                "Lưu kết quả rà soát", type="primary"
            )
        if review_submitted:
            try:
                reviewed = DocumentMetadata.model_validate_json(reviewed_json)
                with st.status(
                    "Đang lưu metadata và tái tạo hai bộ chỉ mục...",
                    expanded=True,
                ) as review_status_view:
                    save_metadata_review_and_reindex(
                        reviewed,
                        reviewer=reviewer,
                        status=review_status,
                        notes=review_notes,
                    )
                    review_status_view.update(
                        label="Đã lưu metadata và cập nhật cả hai chỉ mục.",
                        state="complete",
                        expanded=False,
                    )
            except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
                st.error(f"Không thể lưu metadata đã rà soát: {exc}")
            else:
                get_index.clear()
                get_retriever.clear()
                st.rerun()

upload_records = load_upload_registry()
if upload_records:
    with st.expander("Lịch sử tải tài liệu"):
        st.dataframe(
            pd.DataFrame(upload_records)[
                [
                    "filename",
                    "processing_status",
                    "uploaded_at",
                    "document_id",
                    "page_count",
                    "message",
                ]
            ],
            hide_index=True,
        )

if summary_path.exists():
    with st.expander("Trạng thái ingestion"):
        st.json(json.loads(summary_path.read_text(encoding="utf-8")))
