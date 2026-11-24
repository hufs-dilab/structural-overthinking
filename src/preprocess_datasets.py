"""
Preprocess raw benchmark JSONL files into the dataset/ layout used by run_react.

Inputs are expected at the repository root:
  - aime2025.jsonl
  - gpqa_diamond.jsonl
  - mmlu_pro.jsonl
  - openai_humaneval.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


CHOICE_LETTERS = "ABCDEFGHIJ"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def require_text(row: dict[str, Any], field: str, source_id: str) -> str:
    value = clean_text(row.get(field))
    if not value:
        raise ValueError(f"Missing required field {field!r} for {source_id}")
    return value


def normalize_chat_rows(rows: list[dict[str, Any]], id_prefix: str | None = None) -> list[dict[str, Any]]:
    normalized = []
    for index, row in enumerate(rows):
        item_id = row.get("id") or row.get("task_id") or f"{id_prefix or 'row'}_{index}"
        item = {
            "id": item_id,
            "messages": row["messages"],
            "answer": row["answer"],
        }
        if "answer_index" in row:
            item["answer_index"] = row["answer_index"]
        if "choices" in row:
            item["choices"] = row["choices"]
        normalized.append(item)
    return normalized


def normalize_human_eval(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for index, row in enumerate(rows):
        task_id = row.get("task_id") or f"HumanEval/{index}"
        normalized.append(
            {
                "id": task_id,
                "task_id": task_id,
                "prompt": row["prompt"],
                "test": row["test"],
                "entry_point": row["entry_point"],
            }
        )
    return normalized


def gpqa_answer_fields(row: dict[str, Any]) -> tuple[str, str, list[str]]:
    extra_fields = [
        "Extra Revised Question",
        "Extra Revised Correct Answer",
        "Extra Revised Incorrect Answer 1",
        "Extra Revised Incorrect Answer 2",
        "Extra Revised Incorrect Answer 3",
    ]
    use_extra = all(clean_text(row.get(field)) for field in extra_fields)

    if use_extra:
        return (
            clean_text(row["Extra Revised Question"]),
            clean_text(row["Extra Revised Correct Answer"]),
            [
                clean_text(row["Extra Revised Incorrect Answer 1"]),
                clean_text(row["Extra Revised Incorrect Answer 2"]),
                clean_text(row["Extra Revised Incorrect Answer 3"]),
            ],
        )

    source_id = clean_text(row.get("Record ID")) or clean_text(row.get("Canary String")) or "unknown"
    return (
        require_text(row, "Question", source_id),
        require_text(row, "Correct Answer", source_id),
        [
            require_text(row, "Incorrect Answer 1", source_id),
            require_text(row, "Incorrect Answer 2", source_id),
            require_text(row, "Incorrect Answer 3", source_id),
        ],
    )


def format_gpqa_prompt(question: str, choices: list[str]) -> str:
    lines = [
        "Answer the following multiple-choice question.",
        "Think step by step, then output the final answer in the format \"The answer is (X)\".",
        "",
        f"Question: {question}",
        "Options:",
    ]
    for index, choice in enumerate(choices):
        lines.append(f"{CHOICE_LETTERS[index]}. {choice}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def normalize_gpqa(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for index, row in enumerate(rows):
        record_id = clean_text(row.get("Record ID")) or clean_text(row.get("Canary String")) or str(index)
        question, correct_answer, incorrect_answers = gpqa_answer_fields(row)
        choices_with_labels = [(correct_answer, True)] + [(answer, False) for answer in incorrect_answers]

        rng = random.Random(f"gpqa_diamond:{record_id}")
        rng.shuffle(choices_with_labels)

        choices = [choice for choice, _ in choices_with_labels]
        answer_index = next(i for i, (_, is_correct) in enumerate(choices_with_labels) if is_correct)

        normalized.append(
            {
                "id": f"gpqa_diamond_{record_id}",
                "messages": [{"role": "user", "content": format_gpqa_prompt(question, choices)}],
                "choices": choices,
                "answer": correct_answer,
                "answer_index": answer_index,
            }
        )
    return normalized


def convert(raw_dir: Path, out_dir: Path) -> dict[str, int]:
    outputs = {
        "aime2025.jsonl": normalize_chat_rows(read_jsonl(raw_dir / "aime2025.jsonl"), "aime2025"),
        "mmlu_pro.jsonl": normalize_chat_rows(read_jsonl(raw_dir / "mmlu_pro.jsonl"), "mmlu_pro"),
        "human_eval.jsonl": normalize_human_eval(read_jsonl(raw_dir / "openai_humaneval.jsonl")),
    }

    gpqa_rows = normalize_gpqa(read_jsonl(raw_dir / "gpqa_diamond.jsonl"))
    outputs["gpqa_diamond_5shot_cot_formatted.jsonl"] = gpqa_rows
    outputs["gpqa_diamond.jsonl"] = gpqa_rows

    counts = {}
    for filename, rows in outputs.items():
        write_jsonl(out_dir / filename, rows)
        counts[filename] = len(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("."), help="Directory containing raw JSONL files")
    parser.add_argument("--out-dir", type=Path, default=Path("dataset"), help="Output dataset directory")
    args = parser.parse_args()

    counts = convert(raw_dir=args.raw_dir, out_dir=args.out_dir)
    for filename, count in counts.items():
        print(f"Wrote {args.out_dir / filename} ({count} rows)")


if __name__ == "__main__":
    main()
