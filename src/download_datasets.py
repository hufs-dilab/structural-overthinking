"""Download the benchmarks used in the paper from the Hugging Face Hub into dataset/.

    python -m src.download_datasets                 # aime2025, gpqa_diamond, human_eval
    python -m src.download_datasets --only human_eval

GPQA is gated: accept its terms on https://huggingface.co/datasets/Idavidrein/gpqa
and run `hf auth login` first.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import load_dataset

from src.preprocess_datasets import normalize_gpqa, normalize_human_eval, write_jsonl


def build_aime2025() -> list[dict[str, Any]]:
    rows = []
    for config in ("AIME2025-I", "AIME2025-II"):
        rows.extend(load_dataset("opencompass/AIME2025", config, split="test"))
    return [
        {
            "id": f"aime2025_{index}",
            "messages": [{"role": "user", "content": str(row["question"]).strip()}],
            "answer": str(row["answer"]).strip(),
        }
        for index, row in enumerate(rows)
    ]


def build_human_eval() -> list[dict[str, Any]]:
    return normalize_human_eval(list(load_dataset("openai/openai_humaneval", split="test")))


def build_gpqa_diamond() -> list[dict[str, Any]]:
    return normalize_gpqa(list(load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")))


BUILDERS = {
    "aime2025": build_aime2025,
    "gpqa_diamond": build_gpqa_diamond,
    "human_eval": build_human_eval,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--only", choices=sorted(BUILDERS), nargs="+", default=sorted(BUILDERS))
    args = parser.parse_args()

    for name in args.only:
        rows = BUILDERS[name]()
        write_jsonl(args.out_dir / f"{name}.jsonl", rows)
        if name == "gpqa_diamond":
            write_jsonl(args.out_dir / "gpqa_diamond_5shot_cot_formatted.jsonl", rows)
        print(f"Wrote {args.out_dir / f'{name}.jsonl'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
