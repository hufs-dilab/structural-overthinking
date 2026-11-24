"""
Experiment logger for agent runs.

Logs all information including queries, responses, tokens, and metadata
to uniquely named directories.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentLogger:
    """Logger for agent experiment runs."""

    def __init__(
        self,
        model_id: str,
        agent_type: str,
        base_dir: str = "logs",
        experiment_name: Optional[str] = None,
        evaluator=None,
        dataset_subdir: Optional[str] = None,
        log_dir_name: Optional[str] = None,
        tool_mode: Optional[str] = None,
        nowait: bool = False,
    ):
        """
        Initialize experiment logger.

        Args:
            model_id: Model identifier
            agent_type: Type of agent (tool_call, code, etc.)
            base_dir: Base directory for logs
            experiment_name: Optional experiment name
            evaluator: Optional DatasetEvaluator instance for evaluation
            dataset_subdir: Optional subdirectory name for organizing logs by dataset
            log_dir_name: Optional pre-generated log directory name (for sharing across datasets)
            tool_mode: Optional tool mode (tool, mcp, all)
            nowait: Whether nowait mode is enabled
        """
        self.model_id = model_id
        self.agent_type = agent_type
        self.experiment_name = experiment_name
        self.evaluator = evaluator
        self.tool_mode = tool_mode
        self.nowait = nowait

        # Create unique log directory
        if log_dir_name:
            dir_name = log_dir_name
            timestamp = log_dir_name.split("_")[0] + "_" + log_dir_name.split("_")[1]
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_slug = model_id.replace("/", "_").replace(":", "_")
            tool_mode_str = f"_{tool_mode}" if tool_mode else ""
            nowait_str = "_nowait" if nowait else ""

            if experiment_name:
                dir_name = f"{timestamp}_{model_slug}_{agent_type}{tool_mode_str}{nowait_str}_{experiment_name}"
            else:
                dir_name = f"{timestamp}_{model_slug}_{agent_type}{tool_mode_str}{nowait_str}"

        self.log_dir_name = dir_name  # Store for reuse

        # Add dataset subdirectory if specified
        if dataset_subdir:
            self.log_dir = Path(base_dir) / dir_name / dataset_subdir
        else:
            self.log_dir = Path(base_dir) / dir_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metadata
        self.metadata = {
            "model_id": model_id,
            "agent_type": agent_type,
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "log_dir": str(self.log_dir),
            "dataset_subdir": dataset_subdir,
        }

        self.queries = []
        self.current_query_idx = 0
        self.current_query_idx_for_callback = None

        print(f"📁 Logging to: {self.log_dir}")

    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log configuration parameters.

        Args:
            config: Configuration dictionary
        """
        self.metadata["config"] = config

        config_file = self.log_dir / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"✅ Config saved to: {config_file}")

    def log_tools(self, tools: list[Any]) -> None:
        """
        Log tool information.

        Args:
            tools: List of tools used
        """
        tools_info = []
        for tool in tools:
            tool_data = {
                "name": tool.name if hasattr(tool, "name") else str(tool),
                "description": tool.description if hasattr(tool, "description") else "",
            }
            if hasattr(tool, "_original_id"):
                tool_data["original_id"] = tool._original_id
            tools_info.append(tool_data)

        self.metadata["tools"] = tools_info
        self.metadata["num_tools"] = len(tools_info)

        tools_file = self.log_dir / "tools.json"
        with open(tools_file, "w", encoding="utf-8") as f:
            json.dump(tools_info, f, indent=2, ensure_ascii=False)

        print(f"✅ Tools info saved: {len(tools_info)} tools")

    def start_query(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Start logging a new query.

        Args:
            query: The query string
            metadata: Optional metadata dict (e.g., question_id, correct_answer)

        Returns:
            Query index
        """
        query_data = {
            "idx": self.current_query_idx,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "steps": [],
            "final_response": None,
            "total_tokens": None,
            "input_tokens": None,
            "output_tokens": None,
        }

        # Add metadata if provided
        if metadata:
            query_data["metadata"] = metadata

        self.queries.append(query_data)
        idx = self.current_query_idx
        self.current_query_idx += 1
        self.current_query_idx_for_callback = idx

        print(f"\n🔍 Query {idx}: {query}")

        return idx

    def log_step(self, query_idx: int, step_info: Dict[str, Any]) -> None:
        """
        Log a step in the query execution.

        Args:
            query_idx: Query index
            step_info: Step information dictionary
        """
        if query_idx >= len(self.queries):
            return

        self.queries[query_idx]["steps"].append(step_info)

    def log_steps(self, query_idx: int, steps: list[Dict[str, Any]]) -> None:
        """
        Log multiple steps at once (for parallel processing).

        Args:
            query_idx: Query index
            steps: List of step information dictionaries
        """
        if query_idx >= len(self.queries):
            return

        self.queries[query_idx]["steps"].extend(steps)

    def create_step_collector(self) -> tuple[list[Dict[str, Any]], callable]:
        """
        Create a standalone step collector for parallel processing.

        Returns:
            Tuple of (steps list, step_callback function)
        """
        steps = []

        def step_callback(step, agent=None):
            from smolagents.memory import ActionStep

            if not isinstance(step, ActionStep):
                return

            step_info = {"step_number": step.step_number}

            if hasattr(step, "tool_calls") and step.tool_calls:
                tool_calls = []
                for tc in step.tool_calls:
                    tool_calls.append(
                        {
                            "name": tc.name,
                            "arguments": tc.arguments,
                            "id": tc.id if hasattr(tc, "id") else None,
                        }
                    )
                step_info["tool_calls"] = tool_calls

            if hasattr(step, "observations") and step.observations:
                step_info["observations"] = str(step.observations)

            if hasattr(step, "action_output") and step.action_output:
                step_info["action_output"] = str(step.action_output)

            if hasattr(step, "token_usage") and step.token_usage:
                step_info["token_usage"] = {
                    "input_tokens": step.token_usage.input_tokens,
                    "output_tokens": step.token_usage.output_tokens,
                    "total_tokens": step.token_usage.total_tokens,
                }

            if hasattr(step, "timing") and step.timing:
                step_info["timing"] = {"duration": step.timing.duration}

            if hasattr(step, "error") and step.error:
                step_info["error"] = str(step.error)

            steps.append(step_info)

        return steps, step_callback

    def step_callback(self, step, agent=None) -> None:
        """
        Callback function to be called after each agent step.
        Extracts information from ActionStep and logs it.

        Args:
            step: ActionStep or PlanningStep from agent memory
            agent: Agent instance (optional)
        """
        if self.current_query_idx_for_callback is None:
            return

        # Only log ActionStep (not PlanningStep or other types)
        from smolagents.memory import ActionStep

        if not isinstance(step, ActionStep):
            return

        step_info = {
            "step_number": step.step_number,
        }

        # Extract tool calls
        if hasattr(step, "tool_calls") and step.tool_calls:
            tool_calls = []
            for tc in step.tool_calls:
                tool_calls.append(
                    {
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "id": tc.id if hasattr(tc, "id") else None,
                    }
                )
            step_info["tool_calls"] = tool_calls

        # Extract observations (tool outputs)
        if hasattr(step, "observations") and step.observations:
            step_info["observations"] = str(step.observations)

        # Extract action output
        if hasattr(step, "action_output") and step.action_output:
            step_info["action_output"] = str(step.action_output)

        # Extract token usage
        if hasattr(step, "token_usage") and step.token_usage:
            step_info["token_usage"] = {
                "input_tokens": step.token_usage.input_tokens,
                "output_tokens": step.token_usage.output_tokens,
                "total_tokens": step.token_usage.total_tokens,
            }

        # Extract timing
        if hasattr(step, "timing") and step.timing:
            step_info["timing"] = {
                "duration": step.timing.duration,
            }

        # Extract error if any
        if hasattr(step, "error") and step.error:
            step_info["error"] = str(step.error)

        self.log_step(self.current_query_idx_for_callback, step_info)

    def end_query(
        self,
        query_idx: int,
        response: str,
        token_usage: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        End logging for a query.

        Args:
            query_idx: Query index
            response: Final response
            token_usage: Token usage dict with input_tokens, output_tokens, total_tokens
        """
        if query_idx >= len(self.queries):
            return

        self.queries[query_idx]["final_response"] = response
        self.queries[query_idx]["end_timestamp"] = datetime.now().isoformat()

        # Calculate token usage from steps if not provided
        if token_usage is None:
            token_usage = self._infer_token_usage_from_steps(query_idx)

        if token_usage:
            self.queries[query_idx].update(
                {
                    "total_tokens": token_usage.get("total_tokens"),
                    "input_tokens": token_usage.get("input_tokens"),
                    "output_tokens": token_usage.get("output_tokens"),
                }
            )

        # Evaluate correctness if metadata contains ground truth
        self._evaluate_correctness(query_idx)

        print(f"✅ Query {query_idx} completed")

        if token_usage:
            print(
                f"📊 Tokens - Input: {token_usage.get('input_tokens')}, "
                f"Output: {token_usage.get('output_tokens')}, "
                f"Total: {token_usage.get('total_tokens')}"
            )

    def log_error(self, query_idx: int, error: Exception) -> None:
        """
        Log an error that occurred during query execution.

        Args:
            query_idx: Query index
            error: Exception that occurred
        """
        if query_idx >= len(self.queries):
            return

        error_info = {
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.now().isoformat(),
        }

        self.queries[query_idx]["error"] = error_info
        self.queries[query_idx]["status"] = "error"
        self.queries[query_idx]["end_timestamp"] = datetime.now().isoformat()

        token_usage = self._infer_token_usage_from_steps(query_idx)
        if token_usage:
            self.queries[query_idx].update(
                {
                    "total_tokens": token_usage.get("total_tokens"),
                    "input_tokens": token_usage.get("input_tokens"),
                    "output_tokens": token_usage.get("output_tokens"),
                }
            )

        # Mark failed queries as incorrect for evaluation
        self.queries[query_idx]["is_correct"] = False

        print(f"❌ Error logged for query {query_idx}: {error}")
        if token_usage:
            print(
                f"📊 Partial tokens - Input: {token_usage.get('input_tokens')}, "
                f"Output: {token_usage.get('output_tokens')}, "
                f"Total: {token_usage.get('total_tokens')}"
            )

    def _infer_token_usage_from_steps(self, query_idx: int) -> Optional[Dict[str, int]]:
        """Infer query-level token usage from logged steps."""
        if query_idx >= len(self.queries):
            return None

        total_input = 0
        total_output = 0
        total = 0
        for step in self.queries[query_idx]["steps"]:
            if "token_usage" not in step:
                continue
            usage = step["token_usage"]
            total_input += usage.get("input_tokens", 0)
            total_output += usage.get("output_tokens", 0)
            total += usage.get("total_tokens", 0)

        if total_input > 0 or total_output > 0 or total > 0:
            return {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "total_tokens": total or total_input + total_output,
            }
        return None

    def _evaluate_correctness(self, query_idx: int) -> None:
        """
        Evaluate correctness of the response based on ground truth.

        Args:
            query_idx: Query index
        """
        if self.evaluator is None:
            return

        query_data = self.queries[query_idx]
        metadata = query_data.get("metadata", {})
        response = query_data.get("final_response", "")

        # Extract correct answer from metadata
        correct_answer = metadata.get("correct_answer")

        # For HumanEval, we don't need correct_answer - check for test/prompt/entry_point instead
        has_human_eval_metadata = all(metadata.get(key) for key in ["prompt", "test", "entry_point"])

        if correct_answer is None and not has_human_eval_metadata:
            return

        # Delegate to evaluator
        try:
            eval_result = self.evaluator.evaluate(response, correct_answer, metadata)
            query_data["is_correct"] = eval_result.get("is_correct")
            query_data["predicted_answer"] = eval_result.get("predicted_answer")

            # Include additional fields if present (e.g., result for HumanEval)
            if "result" in eval_result:
                query_data["evaluation_result"] = eval_result["result"]
            if "needs_code_evaluation" in eval_result:
                query_data["needs_code_evaluation"] = eval_result["needs_code_evaluation"]
        except Exception as e:
            print(f"⚠️  Evaluation failed for query {query_idx}: {e}")
            query_data["evaluation_error"] = str(e)


    def finalize(self) -> Path:
        """
        Finalize logging and save summary.


        Returns:
            Path to summary file
        """
        # Calculate summary statistics
        total_queries = len(self.queries)
        successful_queries = sum(1 for q in self.queries if q.get("status") != "error")
        failed_queries = total_queries - successful_queries

        total_tokens = sum(q.get("total_tokens", 0) for q in self.queries if q.get("total_tokens"))
        total_input_tokens = sum(q.get("input_tokens", 0) for q in self.queries if q.get("input_tokens"))
        total_output_tokens = sum(q.get("output_tokens", 0) for q in self.queries if q.get("output_tokens"))

        # Calculate accuracy - include failed queries as incorrect
        # Only count queries that have ground truth (correct_answer in metadata or HumanEval metadata)
        evaluated_queries = [
            q
            for q in self.queries
            if q.get("metadata", {}).get("correct_answer") is not None
            or all(q.get("metadata", {}).get(key) for key in ["prompt", "test", "entry_point"])
        ]
        if evaluated_queries:
            # Count correct: must have is_correct=True AND no error
            correct_count = sum(1 for q in evaluated_queries if q.get("is_correct") == True)
            accuracy = correct_count / len(evaluated_queries) if evaluated_queries else 0.0
        else:
            correct_count = 0
            accuracy = None

        summary = {
            **self.metadata,
            "summary": {
                "total_queries": total_queries,
                "successful_queries": successful_queries,
                "failed_queries": failed_queries,
                "total_tokens": total_tokens,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "evaluated_queries": len(evaluated_queries) if evaluated_queries else 0,
                "correct_count": correct_count,
                "accuracy": accuracy,
            },
            "finalized_at": datetime.now().isoformat(),
        }

        # Save summary (without full queries to keep it lightweight)
        summary_file = self.log_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Save all results in a single file
        all_results_file = self.log_dir / "all_results.json"
        with open(all_results_file, "w", encoding="utf-8") as f:
            json.dump(self.queries, f, indent=2, ensure_ascii=False)
        print(f"✅ All results saved to: {all_results_file}")

        # Also save human-readable report
        report_file = self.log_dir / "report.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("EXPERIMENT REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Model: {self.model_id}\n")
            f.write(f"Agent Type: {self.agent_type}\n")
            f.write(f"Timestamp: {self.metadata['timestamp']}\n")
            if self.experiment_name:
                f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Log Directory: {self.log_dir}\n\n")

            f.write("-" * 80 + "\n")
            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Queries: {total_queries}\n")
            f.write(f"Successful: {successful_queries}\n")
            f.write(f"Failed: {failed_queries}\n")

            if accuracy is not None:
                f.write(f"\nEvaluation Results:\n")
                f.write(f"Evaluated Queries: {len(evaluated_queries)}\n")
                f.write(f"Correct: {correct_count}\n")
                f.write(f"Accuracy: {accuracy:.2%}\n")

            f.write(f"\nToken Usage:\n")
            f.write(f"Total Tokens: {total_tokens:,}\n")
            f.write(f"Input Tokens: {total_input_tokens:,}\n")
            f.write(f"Output Tokens: {total_output_tokens:,}\n\n")

            if self.metadata.get("num_tools"):
                f.write(f"Number of Tools: {self.metadata['num_tools']}\n\n")

            f.write("-" * 80 + "\n")
            f.write("QUERIES\n")
            f.write("-" * 80 + "\n\n")

            for query in self.queries:
                f.write(f"Query {query['idx']}:\n")
                f.write(f"  Query: {query['query']}\n")
                f.write(f"  Status: {query.get('status', 'success')}\n")

                if query.get("error"):
                    f.write(f"  Error: {query['error']['message']}\n")
                else:
                    response = query.get("final_response", "")
                    if len(response) > 200:
                        response = response[:200] + "..."
                    f.write(f"  Response: {response}\n")

                    if query.get("total_tokens"):
                        f.write(f"  Tokens: {query['total_tokens']:,}\n")

                f.write("\n")

        print(f"\n{'=' * 80}")
        print(f"📊 EXPERIMENT SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total Queries: {total_queries}")
        print(f"Successful: {successful_queries}")
        print(f"Failed: {failed_queries}")

        if accuracy is not None:
            print(f"\n🎯 Evaluation Results:")
            print(f"Evaluated: {len(evaluated_queries)}")
            print(f"Correct: {correct_count}")
            print(f"Accuracy: {accuracy:.2%}")

        print(f"\n📊 Token Usage:")
        print(f"Total Tokens: {total_tokens:,}")
        print(f"Input Tokens: {total_input_tokens:,}")
        print(f"Output Tokens: {total_output_tokens:,}")
        print(f"\n✅ Summary saved to: {summary_file}")
        print(f"📄 Report saved to: {report_file}")
        print(f"{'=' * 80}\n")

        return summary_file
