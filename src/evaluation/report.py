"""Run retrieval experiments only on verified questions and export real metrics."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt

from src.config import settings
from src.evaluation.answer_evaluator import create_manual_template
from src.evaluation.case_analysis import write_case_records
from src.evaluation.dataset import is_verified, load_questions
from src.evaluation.retrieval_metrics import (
    hit_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from src.indexing.index_manager import load_strategy
from src.retrieval.hybrid_retriever import HybridRetriever

LOGGER = logging.getLogger(__name__)


def _ids(value: Any) -> set[str]:
    return {item.strip() for item in str(value).split("|") if item.strip()}


def run_evaluation() -> dict[str, Any]:
    questions = load_questions(settings.evaluation_file)
    verified = questions[questions.apply(is_verified, axis=1)]
    if verified.empty:
        summary = {
            "status": "blocked_needs_manual_ground_truth",
            "verified_questions": 0,
            "required_questions": len(questions),
            "message": "Không có câu hỏi được xác minh; không tạo metrics thực nghiệm.",
        }
        output = settings.results_dir / "metrics"
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        create_manual_template(
            settings.evaluation_file, output / "answer_evaluation_template.csv"
        )
        write_case_records([], settings.results_dir / "cases")
        LOGGER.warning(summary["message"])
        return summary

    rows: list[dict[str, Any]] = []
    case_inputs: list[dict[str, Any]] = []
    for strategy in ("fixed_size", "recursive"):
        bundle = load_strategy(strategy)
        retriever = HybridRetriever(bundle)
        for _, question in verified.iterrows():
            strategy_field = f"relevant_chunk_ids_{strategy}"
            relevant = _ids(question[strategy_field] or question["relevant_chunk_ids"])
            if not relevant:
                continue
            started = perf_counter()
            output = retriever.search(question["question"], top_k=3)
            latency = perf_counter() - started
            retrieved = [entry["chunk"].chunk_id for entry in output["results"]]
            row = {
                "strategy": strategy,
                "question_id": question["question_id"],
                "question_type": question["question_type"],
                "precision_at_3": precision_at_k(retrieved, relevant, 3),
                "recall_at_3": recall_at_k(retrieved, relevant, 3),
                "hit_at_3": hit_at_k(retrieved, relevant, 3),
                "mrr": mean_reciprocal_rank(retrieved, relevant),
                "latency_seconds": latency,
                "retrieved_chunk_ids": "|".join(retrieved),
                "relevant_chunk_ids": "|".join(sorted(relevant)),
            }
            rows.append(row)
            case_inputs.append({
                **row,
                "question": question["question"],
                "ground_truth_answer": question["ground_truth_answer"],
                "bm25": output["bm25"],
                "faiss": output["faiss"],
                "rrf": output["rrf"],
                "top_3": [
                    {
                        "chunk_id": item["chunk"].chunk_id,
                        "document_id": item["chunk"].document_id,
                        "page": item["chunk"].page,
                        "section": item["chunk"].section,
                        "text": item["chunk"].text,
                    }
                    for item in output["results"]
                ],
            })

    output_dir = settings.results_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "retrieval_metrics.csv", index=False, encoding="utf-8-sig")
    grouped = (
        frame.groupby(["strategy", "question_type"])[
            ["precision_at_3", "recall_at_3", "hit_at_3", "mrr", "latency_seconds"]
        ]
        .mean()
        .reset_index()
    )
    grouped.to_csv(output_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    figure_dir = settings.results_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    overall = frame.groupby("strategy")[["precision_at_3", "recall_at_3"]].mean()
    axis = overall.plot.bar(figsize=(8, 5), ylim=(0, 1), rot=0)
    axis.set_ylabel("Điểm trung bình")
    axis.set_title("So sánh Fixed-size và Recursive")
    axis.figure.tight_layout()
    axis.figure.savefig(figure_dir / "chunking_comparison.png", dpi=160)
    plt.close(axis.figure)
    summary = {
        "status": "completed",
        "verified_questions": int(verified.shape[0]),
        "rows": len(rows),
        "comparison": grouped.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_cases(case_inputs, settings.results_dir / "cases")
    create_manual_template(
        settings.evaluation_file, output_dir / "answer_evaluation_template.csv"
    )
    return summary


def _write_cases(inputs: list[dict[str, Any]], output_dir: Path) -> None:
    fixed = {case["question_id"]: case for case in inputs if case["strategy"] == "fixed_size"}
    recursive = {
        case["question_id"]: case for case in inputs if case["strategy"] == "recursive"
    }
    successes = [
        (fixed.get(key) or recursive.get(key))
        for key in fixed.keys() | recursive.keys()
        if (fixed.get(key) or recursive.get(key))["hit_at_3"] == 1
    ]
    failures = [
        (fixed.get(key) or recursive.get(key))
        for key in fixed.keys() | recursive.keys()
        if (fixed.get(key) or recursive.get(key))["hit_at_3"] == 0
    ]
    differences = [
        fixed[key]
        for key in fixed.keys() & recursive.keys()
        if fixed[key]["retrieved_chunk_ids"] != recursive[key]["retrieved_chunk_ids"]
    ]
    selected = successes[:1] + failures[:1] + differences[:1]
    cases = []
    for index, case in enumerate(selected, start=1):
        cases.append({
            "case_id": f"case_{index}",
            "status": "retrieved_analysis_pending",
            **case,
            "analysis_status": "needs_human_review",
            "analysis": None,
            "answer": None,
            "citation": None,
            "recommendation": None,
        })
    write_case_records(cases, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(run_evaluation(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
