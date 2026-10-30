"""Write evidence-backed case records; unrun cases remain explicitly unrun."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_case_records(
    cases: list[dict[str, Any]], output_dir: Path, requested_count: int = 3
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(requested_count):
        record = cases[index] if index < len(cases) else {
            "case_id": f"case_{index + 1}",
            "status": "not_run",
            "reason": "Chưa có đủ câu hỏi/ground truth đã xác minh để lập case thực nghiệm.",
        }
        path = output_dir / f"case_{index + 1}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        paths.append(path)
    return paths
