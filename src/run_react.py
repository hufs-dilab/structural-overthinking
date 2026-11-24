"""
ReAct agent evaluation runner.

Usage:
    python -m src.run_react \
        --dataset_name gpqa_diamond \
        --agent_type tool_call \
        --api_base "http://localhost:8000/v1" \
        --api_key "EMPTY" \
        --model_id "zai-org/GLM-4.6" \
        --num_workers 4 \
        --tool_mode all
"""

from datetime import datetime
from typing import Literal, Optional

import fire
import pandas as pd
from dotenv import load_dotenv

from src.common import (
    DatasetEvaluator,
    DatasetRunner,
    ExperimentLogger,
    ModelConfig,
    load_mcp_tools,
    load_tools_from_jsonl,
    sample_tools_by_category,
)
from src.parsers import get_parser, list_parsers


ALL_DATASETS = (
    "human_eval",
    "gpqa_diamond",
    "aime2025",
)

DatasetName = Literal[
    "all",
    "human_eval",
    "gpqa_diamond",
    "gpqa_diamond_5shot_cot_formatted",
    "aime2025",
    "mmlu_pro",
]

ToolMode = Literal["tool", "mcp", "all"]


def load_tools_by_mode(
    tool_mode: ToolMode,
    tool_data_path: str,
    num_tools_per_category: int,
    max_total_tools: int,
) -> list:
    """
    Load tools based on the specified mode.

    Args:
        tool_mode: "tool" for toolret only, "mcp" for MCP only, "all" for both
        tool_data_path: Path to tool definitions JSONL
        num_tools_per_category: Tools to sample per category (for toolret)
        max_total_tools: Maximum total tools (for toolret)

    Returns:
        List of Tool instances
    """
    tools = []

    if tool_mode in ("tool", "all"):
        all_tools = load_tools_from_jsonl(tool_data_path)
        toolret_tools = sample_tools_by_category(
            all_tools,
            n_per_category=num_tools_per_category,
            max_total=max_total_tools,
        )
        tools.extend(toolret_tools)
        print(f"  📦 Loaded {len(toolret_tools)} tools from ToolRet")

    if tool_mode in ("mcp", "all"):
        mcp_tools = load_mcp_tools()
        tools.extend(mcp_tools)
        print(f"  🔧 Loaded {len(mcp_tools)} tools from MCP servers")

    return tools


def evaluate_dataset(
    dataset_name: str,
    agent_type: str,
    model_config: ModelConfig,
    tool_mode: ToolMode,
    tool_data_path: str,
    num_tools_per_category: int,
    max_total_tools: int,
    experiment_name: str,
    num_workers: int,
    tool_parser: str | None = None,
    dataset_subdir: str | None = None,
    log_dir_name: str | None = None,
) -> str:
    """
    Evaluate a single dataset.

    Returns:
        The log_dir_name used (for sharing across datasets)
    """
    dataset_path = f"dataset/{dataset_name}.jsonl"
    try:
        dataset = pd.read_json(dataset_path, lines=True)
    except FileNotFoundError as exc:
        raise ValueError(f"Dataset not found: {dataset_path}") from exc

    print(f"📊 Loaded dataset: {dataset_name} ({len(dataset)} samples)")

    evaluator = DatasetEvaluator(dataset_name=dataset_name)
    logger = ExperimentLogger(
        model_id=model_config.model_id,
        agent_type=agent_type,
        experiment_name=experiment_name,
        evaluator=evaluator,
        dataset_subdir=dataset_subdir,
        log_dir_name=log_dir_name,
        tool_mode=tool_mode,
        nowait=model_config.nowait,
    )

    logger.log_config(
        {
            "dataset_name": dataset_name,
            "dataset_size": len(dataset),
            "agent_type": agent_type,
            "model_id": model_config.model_id,
            "api_base": model_config.api_base,
            "tool_mode": tool_mode,
            "tool_data_path": tool_data_path,
            "num_tools_per_category": num_tools_per_category,
            "max_total_tools": max_total_tools,
            "num_workers": num_workers,
        }
    )

    tools = load_tools_by_mode(
        tool_mode=tool_mode,
        tool_data_path=tool_data_path,
        num_tools_per_category=num_tools_per_category,
        max_total_tools=max_total_tools,
    )
    print(f"Using {len(tools)} tools for the agent")
    logger.log_tools(tools)

    # Setup tool call parser if specified
    parser = None
    if tool_parser:
        parser = get_parser(name=tool_parser)
        if parser:
            parser.set_tools_schema(tools)
            print(f"📝 Using tool parser: {tool_parser}")
    elif tool_parser is None:
        # Auto-detect parser from model ID
        parser = get_parser(model_id=model_config.model_id)
        if parser:
            parser.set_tools_schema(tools)
            print(f"📝 Auto-detected tool parser: {parser.name}")

    runner = DatasetRunner(
        agent_type=agent_type,
        tools=tools,
        logger=logger,
        model_config=model_config,
        num_workers=num_workers,
        tool_parser=parser,
    )
    runner.run(dataset)

    logger.finalize()
    print(f"\n✅ Evaluation complete! Results saved to: {logger.log_dir}")

    return logger.log_dir_name


def main(
    dataset_name: DatasetName = "gpqa_diamond",
    agent_type: Literal["tool_call", "code"] = "tool_call",
    # Model configuration - falls back to env vars if not provided
    model_id: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    # NoWait configuration
    nowait: bool = False,
    # Generation configuration
    max_tokens: int = 16384,
    # Tool configuration
    tool_mode: ToolMode = "tool",
    tool_data_path: str = "tools/toolret_all_tools.jsonl",
    num_tools_per_category: int = 5,
    max_total_tools: int = 100,
    # Tool call parser (auto-detected if not specified, "none" to disable)
    tool_parser: Optional[str] = None,
    # Experiment configuration
    experiment_name: str = "smolagents_direct",
    num_workers: int = 1,
    # Timeout configuration
    request_timeout: int = 300,
) -> None:
    """
    Main entry point for dataset evaluation.

    Args:
        dataset_name: Dataset to evaluate ("all" for all datasets)
        agent_type: Agent type ("tool_call" or "code")
        model_id: Model ID (default: OPENAI_MODEL_ID / VLLM_MODEL_ID env var)
        api_base: API base URL (default: OPENAI_BASE_URL / VLLM_BASE_URL env var)
        api_key: API key (default: OPENAI_API_KEY / VLLM_API_KEY env var)
        nowait: Enable NoWait mode to suppress hesitation tokens (wait, hmm, but, etc.)
        max_tokens: Maximum tokens to generate per completion (default: 16384)
        tool_mode: Tool loading mode ("tool" for ToolRet, "mcp" for MCP servers, "all" for both)
        tool_data_path: Path to tool definitions JSONL
        num_tools_per_category: Tools to sample per category
        max_total_tools: Maximum total tools
        tool_parser: Tool call parser name (auto-detected from model_id if not specified,
                     use "none" to disable). Available: {list_parsers()}
        experiment_name: Experiment name
        num_workers: Parallel workers count
        request_timeout: Timeout in seconds for each API request (default: 300)
    """
    model_config = ModelConfig.from_args(
        model_id=model_id,
        api_base=api_base,
        api_key=api_key,
        nowait=nowait,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
    )

    if not model_config.model_id:
        raise ValueError("model_id is required (via --model_id or OPENAI_MODEL_ID env var)")

    print(f"🤖 Model: {model_config.model_id}")
    print(f"🔗 API Base: {model_config.api_base}")
    print(f"🛠️  Tool Mode: {tool_mode}")
    if nowait:
        print(f"🚫 NoWait Mode: Enabled")
    print(f"📝 Available parsers: {list_parsers()}")

    # Handle "none" as explicit disable
    effective_parser = None if tool_parser == "none" else tool_parser

    datasets = ALL_DATASETS if dataset_name == "all" else (dataset_name,)

    # Generate shared log directory name for multiple datasets
    shared_log_dir_name = None
    if len(datasets) > 1:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = model_config.model_id.replace("/", "_").replace(":", "_")
        nowait_str = "_nowait" if nowait else ""
        shared_log_dir_name = f"{timestamp}_{model_slug}_{agent_type}_{tool_mode}{nowait_str}_{experiment_name}"

    for ds_name in datasets:
        if len(datasets) > 1:
            print(f"\n{'='*80}\n📊 Evaluating: {ds_name}\n{'='*80}\n")

        shared_log_dir_name = evaluate_dataset(
            dataset_name=ds_name,
            agent_type=agent_type,
            model_config=model_config,
            tool_mode=tool_mode,
            tool_data_path=tool_data_path,
            num_tools_per_category=num_tools_per_category,
            max_total_tools=max_total_tools,
            experiment_name=experiment_name,
            num_workers=num_workers,
            tool_parser=effective_parser,
            dataset_subdir=ds_name if len(datasets) > 1 else None,
            log_dir_name=shared_log_dir_name,
        )

    if len(datasets) > 1:
        print(f"\n{'='*80}\n✅ All datasets complete!\n{'='*80}\n")


if __name__ == "__main__":
    load_dotenv(override=True)
    fire.Fire(main)
