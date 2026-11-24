"""Download ToolRet tool definitions used as the benign tool pool for run_react.

Usage:
    python -m src.download_toolret --output tools/toolret_all_tools.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import concatenate_datasets, load_dataset

TOOLRET_REPO = "mangopy/ToolRet-Tools"
TOOLRET_SUBSETS = ("code", "customized", "web")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("tools/toolret_all_tools.jsonl"))
    args = parser.parse_args()

    tools = concatenate_datasets(
        [load_dataset(TOOLRET_REPO, subset)["tools"] for subset in TOOLRET_SUBSETS]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tools.to_pandas().to_json(args.output, orient="records", lines=True, force_ascii=False)
    print(f"Saved {len(tools)} ToolRet tools to {args.output}")


if __name__ == "__main__":
    main()
