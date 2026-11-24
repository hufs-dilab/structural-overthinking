"""Common utilities for agent experiments."""

__all__ = [
    "DatasetEvaluator",
    "DatasetRunner",
    "ExperimentLogger",
    "ModelConfig",
    "QueryItem",
    "load_mcp_tools",
    "load_tools_from_jsonl",
    "prepare_dataset_items",
    "sample_tools_by_category",
]


def __getattr__(name):
    """Import common utilities lazily so CLI-specific deps stay optional."""
    if name in {"QueryItem", "prepare_dataset_items"}:
        from .data_utils import QueryItem, prepare_dataset_items

        return {"QueryItem": QueryItem, "prepare_dataset_items": prepare_dataset_items}[name]

    if name == "DatasetEvaluator":
        from .evaluator import DatasetEvaluator

        return DatasetEvaluator

    if name == "ExperimentLogger":
        from .logger import ExperimentLogger

        return ExperimentLogger

    if name == "ModelConfig":
        from .model_config import ModelConfig

        return ModelConfig

    if name == "DatasetRunner":
        from .runner import DatasetRunner

        return DatasetRunner

    if name in {"load_mcp_tools", "load_tools_from_jsonl", "sample_tools_by_category"}:
        from .tools import load_mcp_tools, load_tools_from_jsonl, sample_tools_by_category

        return {
            "load_mcp_tools": load_mcp_tools,
            "load_tools_from_jsonl": load_tools_from_jsonl,
            "sample_tools_by_category": sample_tools_by_category,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
