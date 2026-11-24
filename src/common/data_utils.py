"""
Data utilities for dataset processing.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class QueryItem:
    """Represents a single query item from a dataset."""

    idx: int
    query: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def extract_query_from_row(row: pd.Series) -> str:
    """
    Extract query string from a dataset row.

    Handles different dataset formats:
    - messages: Chat format with user/assistant messages
    - question: Direct question field
    - prompt: Prompt field (e.g., HumanEval)

    Args:
        row: DataFrame row

    Returns:
        Query string
    """
    if "messages" in row:
        messages = row["messages"]
        user_message = next(
            (m["content"] for m in messages if m["role"] == "user"),
            None,
        )
        return user_message if user_message else str(row.iloc[0])

    if "question" in row:
        return row["question"]

    if "prompt" in row:
        return row["prompt"]

    # Fallback: use first column
    return str(row.iloc[0])


def extract_metadata_from_row(row: pd.Series, idx: int) -> Dict[str, Any]:
    """
    Extract metadata from a dataset row.

    Args:
        row: DataFrame row
        idx: Row index

    Returns:
        Metadata dictionary
    """
    metadata = {"dataset_index": int(idx)}

    # Direct field mappings
    field_mappings = {
        "id": "id",
        "question_id": "question_id",
        "answer": "correct_answer",
        "correct_answer": "correct_answer",
        "answer_index": "correct_answer_index",
        "correct_answer_index": "correct_answer_index",
        "choices": "choices",
        # HumanEval specific fields
        "prompt": "prompt",
        "test": "test",
        "entry_point": "entry_point",
        "task_id": "task_id",
    }

    for source_field, target_field in field_mappings.items():
        if source_field in row and row[source_field] is not None:
            metadata[target_field] = row[source_field]

    return metadata


def prepare_dataset_items(dataset: pd.DataFrame) -> List[QueryItem]:
    """
    Prepare dataset items for processing.

    Args:
        dataset: DataFrame containing the dataset

    Returns:
        List of QueryItem objects
    """
    items = []
    for idx, row in dataset.iterrows():
        query = extract_query_from_row(row)
        metadata = extract_metadata_from_row(row, idx)
        items.append(QueryItem(idx=idx, query=query, metadata=metadata))
    return items