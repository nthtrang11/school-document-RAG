"""Create a manual answer-evaluation sheet; never fabricate human judgments."""

from __future__ import annotations

from pathlib import Path

from src.evaluation.dataset import load_questions


def create_manual_template(questions_path: Path, output_path: Path) -> Path:
    questions = load_questions(questions_path)
    template = questions[["question_id", "question", "ground_truth_answer"]].copy()
    for field in ("correctness", "completeness", "groundedness", "citation_accuracy", "reviewer_notes"):
        template[field] = ""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path
