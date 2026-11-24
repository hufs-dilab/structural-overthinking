"""
Evaluate HumanEval Claude Code session logs separately from inference.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from src.common import DatasetEvaluator


DEFAULT_REPORT_DIR_NAME = "human_eval_eval"


def evaluate_human_eval_results(
    *,
    results_path: Path,
    report_dir: Path,
    max_workers: int,
) -> Path:
    results = _load_result_items(results_path)
    evaluator = DatasetEvaluator(dataset_name="human_eval")
    report_dir.mkdir(parents=True, exist_ok=True)

    evaluations: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        future_to_idx = {
            executor.submit(_evaluate_one, result, evaluator): idx
            for idx, result in enumerate(results)
        }
        for future in as_completed(future_to_idx):
            evaluation = future.result()
            evaluations.append(evaluation)
            print(
                f"[{len(evaluations)}/{len(results)}] "
                f"{evaluation.get('task_id') or evaluation.get('dataset_session_index')} "
                f"{evaluation.get('status')}"
            )

    evaluations.sort(key=lambda item: item.get("dataset_session_index", 0))
    output_path = report_dir / "human_eval_eval.json"
    _write_json(output_path, evaluations)
    _write_json(report_dir / "summary.json", _summarize_evaluations(evaluations))
    print(f"Wrote HumanEval evaluation to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HumanEval Claude Code results.")
    parser.add_argument("--results_path", type=Path, required=True)
    parser.add_argument(
        "--report_dir",
        type=Path,
        help="Directory for evaluation output. Defaults to <results_dir>/human_eval_eval.",
    )
    parser.add_argument("--max_workers", type=int, default=int(os.environ.get("BATCH_SIZE", "30")))
    args = parser.parse_args()

    report_dir = args.report_dir or args.results_path.parent / DEFAULT_REPORT_DIR_NAME
    evaluate_human_eval_results(
        results_path=args.results_path,
        report_dir=report_dir,
        max_workers=args.max_workers,
    )


def _evaluate_one(result: dict[str, Any], evaluator: DatasetEvaluator) -> dict[str, Any]:
    metadata = result.get("metadata") or {}
    started_at = datetime.now().isoformat()
    start = time.monotonic()
    evaluation = {
        "evaluator": "human_eval_local_python",
        "task_id": metadata.get("task_id"),
        "dataset_session_index": metadata.get("dataset_session_index", metadata.get("dataset_index", 0)),
        "started_at": started_at,
        "is_correct": False,
    }

    if not all(metadata.get(key) for key in ["prompt", "test", "entry_point"]):
        return _finish_eval(evaluation, start, "not_evaluated", "Missing prompt, test, or entry_point.")

    response = _extract_code_response(result.get("final_response") or result.get("response") or "")
    try:
        eval_result = evaluator.evaluate(response, None, metadata)
    except Exception as exc:
        return _finish_eval(evaluation, start, "error", f"{type(exc).__name__}: {exc}")

    evaluation["is_correct"] = bool(eval_result.get("is_correct"))
    evaluation["result"] = eval_result.get("result")
    evaluation["status"] = "passed" if evaluation["is_correct"] else "failed"
    evaluation["duration_seconds"] = round(time.monotonic() - start, 3)
    evaluation["ended_at"] = datetime.now().isoformat()
    return evaluation


def _finish_eval(
    evaluation: dict[str, Any],
    start: float,
    status: str,
    error: str | None,
) -> dict[str, Any]:
    evaluation["status"] = status
    evaluation["duration_seconds"] = round(time.monotonic() - start, 3)
    evaluation["ended_at"] = datetime.now().isoformat()
    if error:
        evaluation["error"] = error
    return evaluation


def _summarize_evaluations(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for evaluation in evaluations:
        status = str(evaluation.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    evaluated_count = sum(1 for item in evaluations if item.get("status") in {"passed", "failed"})
    correct_count = sum(1 for item in evaluations if item.get("is_correct") is True)
    return {
        "evaluator": "human_eval_local_python",
        "total_count": len(evaluations),
        "evaluated_count": evaluated_count,
        "correct_count": correct_count,
        "accuracy": round(correct_count / evaluated_count, 6) if evaluated_count else 0.0,
        "status_counts": status_counts,
    }


def _extract_code_response(response: str) -> str:
    fenced = re.search(r"```(?:python|py)?\s*\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if fenced:
        content = fenced.group(1)
        return content.rstrip() + "\n"
    return response


def _load_result_items(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)

    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("queries", "results"):
            if isinstance(value.get(key), list):
                return value[key]
        if all(isinstance(item, dict) for item in value.values()):
            return list(value.values())

    raise ValueError(f"Unsupported results format: {path}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
