#!/usr/bin/env python3
"""
Log analysis script.

Parses experiment results under logs/ and builds per-model, per-dataset tables.

Usage:
    python src/analyze_logs.py [--logs-dir LOGS_DIR] [--output OUTPUT] [--format {table,csv,json}]

    python3 src/analyze_logs.py --format json --output logs/analysis.json --detailed
    python3 src/analyze_logs.py --format csv --output logs/analysis.csv --detailed
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def parse_experiment_dir_name(dir_name: str) -> Dict[str, str]:
    """Extract metadata from an experiment directory name.

    Example: 20251218_154011_glm-4.6_tool_call_my_experiment
    -> timestamp, model_id, agent_type, experiment_name
    """
    parts = dir_name.split("_")
    if len(parts) >= 5:
        timestamp = f"{parts[0]}_{parts[1]}"
        model_id = parts[2]
        agent_type = parts[3]
        experiment_name = "_".join(parts[4:])
        return {
            "timestamp": timestamp,
            "model_id": model_id,
            "agent_type": agent_type,
            "experiment_name": experiment_name,
        }
    return {"dir_name": dir_name}


def load_summary(summary_path: Path) -> Optional[Dict[str, Any]]:
    """Load a summary.json file."""
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {summary_path}: {e}")
        return None


def collect_all_results(logs_dir: Path) -> List[Dict[str, Any]]:
    """Collect all experiment results."""
    results = []

    if not logs_dir.exists():
        print(f"Error: Logs directory not found: {logs_dir}")
        return results

    # Iterate over experiment directories
    for exp_dir in logs_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        exp_meta = parse_experiment_dir_name(exp_dir.name)

        # Iterate over dataset subdirectories
        for dataset_dir in exp_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset_name = dataset_dir.name
            summary_path = dataset_dir / "summary.json"

            summary = load_summary(summary_path)
            if summary is None:
                continue

            # Build the result record
            result = {
                "experiment_dir": exp_dir.name,
                "dataset": dataset_name,
                "model_id": summary.get("model_id", exp_meta.get("model_id", "unknown")),
                "agent_type": summary.get("agent_type", exp_meta.get("agent_type", "unknown")),
                "experiment_name": summary.get("experiment_name", exp_meta.get("experiment_name", "unknown")),
                "timestamp": summary.get("timestamp", exp_meta.get("timestamp", "unknown")),
            }

            # Add summary statistics
            if "summary" in summary:
                stats = summary["summary"]
                result.update(
                    {
                        "total_queries": stats.get("total_queries", 0),
                        "successful_queries": stats.get("successful_queries", 0),
                        "failed_queries": stats.get("failed_queries", 0),
                        "evaluated_queries": stats.get("evaluated_queries", 0),
                        "correct_count": stats.get("correct_count", 0),
                        "accuracy": stats.get("accuracy", 0.0),
                        "total_tokens": stats.get("total_tokens", 0),
                        "input_tokens": stats.get("total_input_tokens", 0),
                        "output_tokens": stats.get("total_output_tokens", 0),
                    }
                )

            # Extra fields from config
            if "config" in summary:
                config = summary["config"]
                result.update(
                    {
                        "dataset_size": config.get("dataset_size", 0),
                        "num_tools": summary.get("num_tools", config.get("max_total_tools", 0)),
                    }
                )

            results.append(result)

    return results


def create_accuracy_table(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Build a model x dataset accuracy table."""
    table = defaultdict(dict)

    for r in results:
        model = r["model_id"]
        dataset = r["dataset"]
        accuracy = r.get("accuracy", 0.0)
        table[model][dataset] = accuracy

    return dict(table)


def create_token_usage_table(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Build a model x dataset token usage table."""
    table = defaultdict(dict)

    for r in results:
        model = r["model_id"]
        dataset = r["dataset"]
        tokens = r.get("total_tokens", 0)
        table[model][dataset] = tokens

    return dict(table)


def create_success_rate_table(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Build a model x dataset success rate table."""
    table = defaultdict(dict)

    for r in results:
        model = r["model_id"]
        dataset = r["dataset"]
        total = r.get("total_queries", 0)
        success = r.get("successful_queries", 0)
        rate = (success / total * 100) if total > 0 else 0.0
        table[model][dataset] = rate

    return dict(table)


def print_table(table: Dict[str, Dict[str, Any]], title: str, format_str: str = "{:.2f}"):
    """Print a table to the console."""
    print(f"\n{'='*80}")
    print(f" {title}")
    print("=" * 80)

    if not table:
        print("No data available.")
        return

    # Collect all datasets
    all_datasets = set()
    for model_data in table.values():
        all_datasets.update(model_data.keys())
    datasets = sorted(all_datasets)
    models = sorted(table.keys())

    if not datasets or not models:
        print("No data available.")
        return

    # Compute column widths
    model_col_width = max(len("Model"), max(len(m) for m in models))
    col_widths = {d: max(len(d), 12) for d in datasets}

    # Print header
    header = f"{'Model':<{model_col_width}}"
    for d in datasets:
        header += f" | {d:>{col_widths[d]}}"
    print(header)
    print("-" * len(header))

    # Print rows
    for model in models:
        row = f"{model:<{model_col_width}}"
        for d in datasets:
            value = table[model].get(d, None)
            if value is not None:
                if isinstance(value, float):
                    formatted = format_str.format(value)
                else:
                    formatted = str(value)
            else:
                formatted = "-"
            row += f" | {formatted:>{col_widths[d]}}"
        print(row)


def print_detailed_results(results: List[Dict[str, Any]]):
    """Print detailed results."""
    print(f"\n{'='*80}")
    print(" DETAILED RESULTS")
    print("=" * 80)

    for r in sorted(results, key=lambda x: (x["model_id"], x["dataset"])):
        print(f"\n[{r['model_id']}] {r['dataset']}")
        print(f"  Experiment: {r['experiment_name']} ({r['timestamp']})")
        print(f"  Agent Type: {r['agent_type']}")
        print(
            f"  Queries: {r.get('total_queries', 0)} (Success: {r.get('successful_queries', 0)}, Failed: {r.get('failed_queries', 0)})"
        )
        print(f"  Accuracy: {r.get('accuracy', 0):.2%} ({r.get('correct_count', 0)}/{r.get('evaluated_queries', 0)})")
        print(
            f"  Tokens: {r.get('total_tokens', 0):,} (In: {r.get('input_tokens', 0):,}, Out: {r.get('output_tokens', 0):,})"
        )


def export_to_csv(results: List[Dict[str, Any]], output_path: Path):
    """Export results to CSV."""
    if HAS_PANDAS:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Results exported to: {output_path}")
    else:
        import csv

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        print(f"Results exported to: {output_path}")


def export_to_json(results: List[Dict[str, Any]], output_path: Path):
    """Export results to JSON."""
    output = {
        "generated_at": datetime.now().isoformat(),
        "total_experiments": len(results),
        "results": results,
        "tables": {
            "accuracy": create_accuracy_table(results),
            "token_usage": create_token_usage_table(results),
            "success_rate": create_success_rate_table(results),
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results exported to: {output_path}")


def create_pandas_pivot_tables(results: List[Dict[str, Any]]) -> Dict[str, "pd.DataFrame"]:
    """Build pivot tables with pandas."""
    if not HAS_PANDAS:
        return {}

    df = pd.DataFrame(results)

    tables = {}

    # Accuracy pivot table
    if "accuracy" in df.columns:
        tables["accuracy"] = df.pivot_table(values="accuracy", index="model_id", columns="dataset", aggfunc="mean")

    # Token usage pivot table
    if "total_tokens" in df.columns:
        tables["total_tokens"] = df.pivot_table(
            values="total_tokens", index="model_id", columns="dataset", aggfunc="sum"
        )

    # Average tokens per query
    if "total_tokens" in df.columns and "total_queries" in df.columns:
        df["tokens_per_query"] = df["total_tokens"] / df["total_queries"].replace(0, 1)
        tables["tokens_per_query"] = df.pivot_table(
            values="tokens_per_query", index="model_id", columns="dataset", aggfunc="mean"
        )

    return tables


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment logs")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Path to logs directory")
    parser.add_argument("--output", type=str, default=None, help="Output file path (csv or json)")
    parser.add_argument("--format", choices=["table", "csv", "json"], default="table", help="Output format")
    parser.add_argument("--detailed", action="store_true", help="Show detailed results")

    args = parser.parse_args()

    # Resolve the logs directory
    logs_dir = Path(args.logs_dir)
    if not logs_dir.is_absolute():
        # Resolve relative paths against the script location
        script_dir = Path(__file__).parent.parent
        logs_dir = script_dir / args.logs_dir

    print(f"Analyzing logs from: {logs_dir}")

    # Collect results
    results = collect_all_results(logs_dir)

    if not results:
        print("No results found.")
        return

    print(f"Found {len(results)} experiment results.")

    # Dispatch on output format
    if args.format == "csv":
        output_path = Path(args.output) if args.output else logs_dir / "analysis_results.csv"
        export_to_csv(results, output_path)
    elif args.format == "json":
        output_path = Path(args.output) if args.output else logs_dir / "analysis_results.json"
        export_to_json(results, output_path)
    else:
        # Table output
        print(f"\n{'#'*80}")
        print(f"# LOG ANALYSIS REPORT")
        print(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*80}")

        # Accuracy table
        accuracy_table = create_accuracy_table(results)
        print_table(accuracy_table, "ACCURACY BY MODEL x DATASET", "{:.2%}")

        # Success rate table
        success_table = create_success_rate_table(results)
        print_table(success_table, "SUCCESS RATE (%) BY MODEL x DATASET", "{:.1f}%")

        # Token usage table
        token_table = create_token_usage_table(results)
        print_table(token_table, "TOTAL TOKENS BY MODEL x DATASET", "{:,.0f}")

        # Extra analysis when pandas is available
        if HAS_PANDAS:
            pivot_tables = create_pandas_pivot_tables(results)
            if "tokens_per_query" in pivot_tables:
                print(f"\n{'='*80}")
                print(" AVERAGE TOKENS PER QUERY")
                print("=" * 80)
                print(pivot_tables["tokens_per_query"].to_string(float_format="{:,.0f}".format))

        # Detailed results
        if args.detailed:
            print_detailed_results(results)

        # Summary
        print(f"\n{'='*80}")
        print(" SUMMARY")
        print("=" * 80)
        models = set(r["model_id"] for r in results)
        datasets = set(r["dataset"] for r in results)
        print(f"  Models: {', '.join(sorted(models))}")
        print(f"  Datasets: {', '.join(sorted(datasets))}")
        print(f"  Total Experiments: {len(results)}")

        # Overall average accuracy
        accuracies = [r["accuracy"] for r in results if "accuracy" in r]
        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            print(f"  Average Accuracy: {avg_accuracy:.2%}")


if __name__ == "__main__":
    main()
