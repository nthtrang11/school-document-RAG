"""Run retrieval experiments only on manually verified ground truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.config import settings
from src.evaluation.answer_evaluator import create_manual_template
from src.evaluation.case_analysis import write_case_records
from src.evaluation.dataset import (
    is_verified,
    load_questions,
    query_type,
    question_skip_reason,
    relevance_scores,
    relevant_document_ids,
)
from src.evaluation.retrieval_metrics import (
    hit_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.indexing.index_manager import load_strategy
from src.retrieval.hybrid_retriever import HybridRetriever

LOGGER = logging.getLogger(__name__)
_METHODS = ("bm25", "faiss", "hybrid")
_METRICS = (
    "precision_at_3",
    "recall_at_3",
    "hit_at_3",
    "mrr",
    "ndcg_at_3",
    "latency_seconds",
)


def _dataset_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    def make_json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: make_json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [make_json_safe(item) for item in value]
        if hasattr(value, "item"):
            return value.item()
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(make_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _unique_documents(
    chunk_ids: list[str], document_by_chunk: dict[str, str], limit: int = 3
) -> list[str]:
    documents = []
    seen = set()
    for chunk_id in chunk_ids:
        document_id = document_by_chunk.get(chunk_id)
        if document_id and document_id not in seen:
            seen.add(document_id)
            documents.append(document_id)
            if len(documents) == limit:
                break
    return documents


def _aggregate(frame: pd.DataFrame, fields: list[str]) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    grouped = (
        frame.groupby(fields, dropna=False)[list(_METRICS)]
        .agg(["mean", "median", "count"])
    )
    grouped.columns = [
        "_".join(str(part) for part in column if part)
        for column in grouped.columns.to_flat_index()
    ]
    return grouped.reset_index().to_dict(orient="records")


def _method_rankings(
    retrieval: dict[str, Any], document_by_chunk: dict[str, str]
) -> dict[str, list[str]]:
    return {
        "bm25": _unique_documents(
            [chunk_id for chunk_id, _ in retrieval["bm25"]], document_by_chunk
        ),
        "faiss": _unique_documents(
            [chunk_id for chunk_id, _ in retrieval["faiss"]], document_by_chunk
        ),
        "hybrid": _unique_documents(
            [
                item["chunk"].chunk_id
                for item in retrieval["results"][:3]
            ],
            document_by_chunk,
        ),
    }


def _top_results(
    retrieval: dict[str, Any],
    method: str,
    chunk_by_id: dict[str, Any],
) -> list[dict[str, Any]]:
    if method == "bm25":
        chunk_ids = [chunk_id for chunk_id, _ in retrieval["bm25"][:3]]
        scores = dict(retrieval["bm25"])
        return [
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "score": scores.get(chunk_id),
                "document_id": chunk_by_id[chunk_id].document_id,
                "page": chunk_by_id[chunk_id].page,
                "text": chunk_by_id[chunk_id].text,
            }
            for rank, chunk_id in enumerate(chunk_ids, start=1)
        ]
    if method == "faiss":
        chunk_ids = [chunk_id for chunk_id, _ in retrieval["faiss"][:3]]
        scores = dict(retrieval["faiss"])
        return [
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "score": scores.get(chunk_id),
                "document_id": chunk_by_id[chunk_id].document_id,
                "page": chunk_by_id[chunk_id].page,
                "text": chunk_by_id[chunk_id].text,
            }
            for rank, chunk_id in enumerate(chunk_ids, start=1)
        ]
    return [
        {
            "rank": rank,
            "chunk_id": item["chunk"].chunk_id,
            "score": item.get("rrf_score"),
            "document_id": item["chunk"].document_id,
            "page": item["chunk"].page,
            "text": item["chunk"].text,
        }
        for rank, item in enumerate(retrieval["results"][:3], start=1)
    ]


def _skip_records(
    questions: pd.DataFrame, verified: pd.DataFrame
) -> list[dict[str, str]]:
    valid_ids = set(verified["question_id"].astype(str))
    duplicate_ids = set(
        questions.loc[
            questions["question_id"].astype(str).duplicated(keep=False),
            "question_id",
        ].astype(str)
    )
    skipped = []
    for _, question in questions.iterrows():
        question_id = str(question.get("question_id", ""))
        if question_id in valid_ids and question_id not in duplicate_ids:
            continue
        reason = (
            "ground_truth_invalid:duplicate_question_id"
            if question_id in duplicate_ids
            else question_skip_reason(question) or "ground_truth_invalid"
        )
        skipped.append({"question_id": question_id, "reason": reason})
    return skipped


def _write_case_studies(
    cases: list[dict[str, Any]], output_dir: Path
) -> None:
    by_kind: dict[str, dict[str, Any]] = {}
    for case in cases:
        if case["method"] != "hybrid":
            continue
        if case["hit_at_3"] == 0:
            by_kind.setdefault("retrieval_failure", case)
        elif case["mrr"] < 1.0 or case["recall_at_3"] < 1.0:
            by_kind.setdefault("retrieval_suboptimal", case)
        else:
            by_kind.setdefault("retrieval_success", case)

    selected = []
    for index, kind in enumerate(
        ("retrieval_success", "retrieval_failure", "retrieval_suboptimal"),
        start=1,
    ):
        case = by_kind.get(kind)
        if case is None:
            selected.append(
                {
                    "case_id": f"case_{index}",
                    "status": "not_run",
                    "case_type": kind,
                    "reason": (
                        "Không có kết quả thực nghiệm đã xác minh thuộc nhóm này."
                    ),
                }
            )
            continue
        relevant = set(case["relevant_document_ids"])
        ranks = case["method_document_rankings"]
        if kind == "retrieval_success":
            analysis = "Tài liệu Ground Truth xuất hiện trong Top 3 Hybrid."
            recommendation = "Kiểm tra bằng chứng và độ ổn định qua các lần chạy."
        elif kind == "retrieval_failure":
            analysis = "Không có tài liệu Ground Truth trong Top 3 Hybrid."
            recommendation = (
                "Đối chiếu thứ hạng BM25 và FAISS để xác định nguồn lỗi trước khi "
                "điều chỉnh retrieval."
            )
        else:
            analysis = (
                "Có tài liệu Ground Truth trong Top 3 nhưng thứ hạng/độ bao phủ "
                "chưa tối ưu."
            )
            recommendation = (
                "So sánh thứ hạng BM25, FAISS và Hybrid; review thủ công trước khi "
                "đổi cấu hình."
            )
        selected.append(
            {
                "case_id": f"case_{index}",
                "status": "retrieved_analysis_pending",
                "case_type": kind,
                **case,
                "relevant_document_ids": sorted(relevant),
                "method_document_rankings": ranks,
                "analysis_status": "needs_human_review",
                "analysis": analysis,
                "recommendation": recommendation,
                "answer": None,
                "citation": None,
            }
        )
    write_case_records(selected, output_dir)


def run_evaluation() -> dict[str, Any]:
    questions = load_questions(settings.evaluation_file)
    verified = questions[questions.apply(is_verified, axis=1)].copy()
    duplicate_ids = questions["question_id"].astype(str).duplicated(keep=False)
    verified = verified.loc[~duplicate_ids.loc[verified.index]].copy()
    skipped = _skip_records(questions, verified)
    timestamp = datetime.now(timezone.utc).isoformat()
    output_dir = settings.results_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    case_inputs: list[dict[str, Any]] = []
    if not verified.empty:
        for strategy in ("fixed_size", "recursive"):
            bundle = load_strategy(strategy)
            retriever = HybridRetriever(bundle)
            document_by_chunk = {
                chunk.chunk_id: chunk.document_id for chunk in bundle.chunks
            }
            chunk_by_id = {chunk.chunk_id: chunk for chunk in bundle.chunks}
            for _, question in verified.iterrows():
                relevant = relevant_document_ids(question)
                graded = relevance_scores(question)
                started = perf_counter()
                retrieval = retriever.search(str(question["question"]), top_k=3)
                elapsed = perf_counter() - started
                rankings = _method_rankings(retrieval, document_by_chunk)
                latencies = retrieval.get("latency_seconds", {})
                for method in _METHODS:
                    ranked_documents = rankings[method]
                    metric_row = {
                        "strategy": strategy,
                        "method": method,
                        "question_id": str(question["question_id"]),
                        "question_type": query_type(question),
                        "evaluation_unit": "document",
                        "precision_at_3": precision_at_k(
                            ranked_documents, relevant, 3
                        ),
                        "recall_at_3": recall_at_k(ranked_documents, relevant, 3),
                        "hit_at_3": hit_at_k(ranked_documents, relevant, 3),
                        "mrr": mean_reciprocal_rank(ranked_documents, relevant),
                        "ndcg_at_3": ndcg_at_k(ranked_documents, graded, 3),
                        "latency_seconds": float(latencies.get(method, elapsed)),
                        "retrieved_document_ids": "|".join(ranked_documents),
                        "relevant_document_ids": "|".join(sorted(relevant)),
                        "retrieved_chunk_ids": "|".join(
                            chunk_id
                            for chunk_id, _ in retrieval.get(method, [])
                        )
                        if method != "hybrid"
                        else "|".join(
                            item["chunk"].chunk_id
                            for item in retrieval["results"][:3]
                        ),
                    }
                    rows.append(metric_row)
                    if method == "hybrid":
                        case_inputs.append(
                            {
                                **metric_row,
                                "question": str(question["question"]),
                                "ground_truth_answer": str(
                                    question.get("ground_truth_answer", "")
                                ),
                                "method_document_rankings": rankings,
                                "method_results": {
                                    key: _top_results(retrieval, key, chunk_by_id)
                                    for key in _METHODS
                                },
                            }
                        )

    metrics_path = output_dir / "retrieval_metrics.csv"
    comparison_path = output_dir / "comparison.csv"
    skipped_path = output_dir / "skipped_questions.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(skipped, columns=["question_id", "reason"]).to_csv(
        skipped_path, index=False, encoding="utf-8-sig"
    )
    by_query_type = _aggregate(frame, ["strategy", "method", "question_type"])
    comparison = _aggregate(frame, ["strategy", "method"])
    pd.DataFrame(comparison).to_csv(
        comparison_path, index=False, encoding="utf-8-sig"
    )

    if not frame.empty:
        figure_dir = settings.results_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        plot_data = frame.groupby(["strategy", "method"])[
            ["precision_at_3", "recall_at_3", "hit_at_3", "mrr", "ndcg_at_3"]
        ].mean()
        axis = plot_data.plot.bar(figsize=(11, 5), ylim=(0, 1), rot=25)
        axis.set_ylabel("Điểm trung bình")
        axis.set_title("So sánh retrieval theo chunking và phương pháp")
        axis.figure.tight_layout()
        axis.figure.savefig(
            figure_dir / "retrieval_comparison.png", dpi=160
        )
        plt.close(axis.figure)

    _write_case_studies(case_inputs, settings.results_dir / "cases")
    create_manual_template(
        settings.evaluation_file, output_dir / "answer_evaluation_template.csv"
    )

    minimum = settings.evaluation_min_verified_questions
    if len(verified) == 0:
        status = "blocked_needs_manual_ground_truth"
    elif len(verified) < minimum:
        status = "insufficient_verified_ground_truth"
    else:
        status = "completed"
    summary = {
        "status": status,
        "run_at": timestamp,
        "dataset_path": str(settings.evaluation_file),
        "dataset_sha256": _dataset_digest(settings.evaluation_file),
        "required_verified_questions": minimum,
        "total_questions": int(len(questions)),
        "verified_questions": int(len(verified)),
        "evaluated_rows": len(rows),
        "skipped_questions": len(skipped),
        "skip_reasons": {
            reason: sum(item["reason"] == reason for item in skipped)
            for reason in sorted({item["reason"] for item in skipped})
        },
        "evaluation_unit": "document",
        "top_k": 3,
        "strategies": ["fixed_size", "recursive"],
        "methods": list(_METHODS),
        "retrieval_config": {
            "bm25_top_k": settings.bm25_top_k,
            "faiss_top_k": settings.faiss_top_k,
            "rrf_k": settings.rrf_k,
            "final_top_k": settings.final_top_k,
            "embedding_model": settings.embedding_model,
        },
        "comparison": comparison,
        "by_query_type": by_query_type,
        "artifacts": {
            "metrics_csv": str(metrics_path),
            "comparison_csv": str(comparison_path),
            "skipped_questions_csv": str(skipped_path),
            "case_studies": str(settings.results_dir / "cases"),
        },
    }
    if status != "completed":
        summary["message"] = (
            f"Cần tối thiểu {minimum} câu hỏi có Ground Truth hợp lệ đã xác minh; "
            f"hiện có {len(verified)}."
        )
        LOGGER.warning(summary["message"])
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(run_evaluation(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
