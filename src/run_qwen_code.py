"""
Qwen Code runner for HumanEval with OpenAI-compatible models.

Usage:
    python -m src.run_qwen_code \
        --dataset_path dataset/human_eval.jsonl \
        --model_id my-model \
        --registry_mode mixed \
        --num_workers 4
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import fire
from dotenv import load_dotenv
from tqdm import tqdm

from src.common.agent_cli import (
    ATTACK_MCP_SERVER_NAMES,
    CONTEXT7_PACKAGE,
    build_humaneval_prompt,
    compact_stream_metrics,
    count_tool_uses,
    extract_stream_metrics,
    extract_stream_text,
    extract_text_from_json_stdout,
    item_slug,
    load_humaneval_items,
    normalize_humaneval_completion,
    prepare_item_workspace,
    resolve_mcp_python,
    resolve_required_executable,
    resolve_serena_executable,
    run_agent_subprocess,
    start_run_local_mcp_servers,
    stop_run_local_mcp_servers,
    token_usage_from_mapping,
    write_json_file,
    write_jsonl_file,
    write_text_file,
)
from src.common.data_utils import QueryItem
from src.common.evaluator import DatasetEvaluator
from src.common.logger import ExperimentLogger
from src.common.nowait_tokens import get_nowait_logit_bias


REGISTRY_MODES = {"normal", "mixed"}
QWEN_MCP_HTTP_TIMEOUT_MS = 30000
QWEN_PROVIDER_API_KEY_ENV = "OPENAI_API_KEY"


def coerce_bool(value: Any, *, name: str) -> bool:
    """Normalize bool-like CLI values passed by Fire."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"{name} must be a boolean value")


@dataclass
class QwenCodeResult:
    """Result from one Qwen Code HumanEval invocation."""

    idx: int
    query: str
    metadata: dict[str, Any]
    response: Optional[str]
    steps: list[dict[str, Any]]
    error: Optional[Exception]
    trace_dir: Optional[str] = None
    workspace_dir: Optional[str] = None


def normalize_openai_base_url(base_url: str) -> str:
    """Normalize an OpenAI-compatible base URL for Qwen Code."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def resolve_qwen_api_key(api_key_env: str = "OPENAI_API_KEY") -> str:
    """Load the OpenAI-compatible provider API key."""
    api_key = os.getenv(api_key_env) or os.getenv("VLLM_API_KEY")
    if not api_key:
        raise ValueError(f"{api_key_env} (or VLLM_API_KEY) is required")
    return api_key


def resolve_qwen_base_url(base_url: Optional[str]) -> str:
    """Resolve the OpenAI-compatible model endpoint."""
    resolved = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("VLLM_BASE_URL")
    if not resolved:
        raise ValueError("--base_url, OPENAI_BASE_URL, or VLLM_BASE_URL is required")
    return resolved


def build_nowait_extra_body(model_id: str, nowait: bool) -> dict[str, Any]:
    """Build OpenAI-compatible request-body overrides for NoWait mode."""
    if not nowait:
        return {}
    logit_bias = get_nowait_logit_bias(model_id)
    if not logit_bias:
        return {}
    return {"logit_bias": logit_bias}


def extract_text_from_qwen_stdout(stdout: str) -> tuple[str, Optional[Any]]:
    """Extract the textual answer from Qwen Code stdout."""
    stripped = stdout.strip()
    if not stripped:
        return "", None

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped, None

    if isinstance(payload, list):
        events = [event for event in payload if isinstance(event, dict)]
        if events:
            return extract_stream_text(events), payload
        return stripped, payload

    if isinstance(payload, dict):
        return extract_text_from_json_stdout(stdout)

    return stripped, payload


def parse_qwen_output(stdout: str) -> tuple[str, dict[str, Any]]:
    """Parse Qwen Code json or stream-json output."""
    stripped = stdout.strip()
    if not stripped:
        return "", {"format": "empty"}

    events: list[dict[str, Any]] = []
    parsed_json_lines = True
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            parsed_json_lines = False
            events = []
            break
        if isinstance(payload, dict):
            events.append(payload)
        elif isinstance(payload, list) and len(stripped.splitlines()) == 1:
            events.extend(event for event in payload if isinstance(event, dict))
        else:
            parsed_json_lines = False
            events = []
            break

    if parsed_json_lines and events:
        metrics = extract_stream_metrics(events)
        return extract_stream_text(events), {
            "format": "stream-json" if len(stripped.splitlines()) > 1 else "json",
            "events": events,
            "metrics": metrics,
        }

    text, payload = extract_text_from_qwen_stdout(stdout)
    if isinstance(payload, list):
        events = [event for event in payload if isinstance(event, dict)]
        metrics = extract_stream_metrics(events)
        return text, {
            "format": "json",
            "events": events,
            "metrics": metrics,
        }

    usage = token_usage_from_mapping(payload.get("usage") if isinstance(payload, dict) else None)
    return text, {
        "format": "json" if payload is not None else "text",
        "payload": payload,
        "metrics": {
            "num_turns": int(payload.get("num_turns") or 0) if isinstance(payload, dict) else 0,
            "tokens": usage,
            "turns": [],
            "tool_call_count": count_tool_uses(payload),
            "max_tool_call_depth_observed": 0,
            "available_tools": 0,
            "available_tool_names": [],
            "mcp_servers": [],
            "api_retries": 0,
            "duration_ms": payload.get("duration_ms") if isinstance(payload, dict) else None,
            "duration_api_ms": payload.get("duration_api_ms") if isinstance(payload, dict) else None,
            "total_cost_usd": float(payload.get("total_cost_usd") or 0) if isinstance(payload, dict) else 0.0,
            "permission_denials": payload.get("permission_denials", []) if isinstance(payload, dict) else [],
            "model_usage": payload.get("modelUsage", {}) if isinstance(payload, dict) else {},
            "result_event": payload if isinstance(payload, dict) else {},
            "init_event": {},
        },
    }


def build_qwen_humaneval_prompt(
    item: QueryItem,
    *,
    registry_mode: str,
    system_prompt: str,
) -> str:
    """Build a Qwen-compatible prompt; Qwen Code has no append-system-prompt flag."""
    prompt = build_humaneval_prompt(item, registry_mode=registry_mode)
    if not system_prompt:
        return prompt
    return f"{system_prompt}\n\n{prompt}"


def build_qwen_normal_mcp_server_specs(
    *,
    project_dir: Any,
    serena_command: Optional[str],
    serena_python: Optional[str],
    serena_context: str,
    context7_npx: Optional[str],
    mcp_env: Optional[dict[str, str]] = None,
) -> list[dict[str, Any]]:
    """Build run-local normal registry MCP server specs for one Qwen workspace."""
    project_path = str(project_dir)
    serena_command = resolve_serena_executable(
        serena_command=serena_command,
        serena_python=serena_python,
    )
    npx_command = resolve_required_executable("npx", context7_npx)

    serena_args: list[str] = [
        "start-mcp-server",
        "--context",
        serena_context,
        "--project",
        project_path,
        "--enable-web-dashboard",
        "false",
        "--enable-gui-log-window",
        "false",
    ]

    return [
        {
            "name": "serena",
            "type": "stdio",
            "command": serena_command,
            "args": serena_args,
            "scope": "item-local",
            "project_dir": project_path,
            "env": mcp_env or {},
        },
        {
            "name": "context7",
            "type": "stdio",
            "command": npx_command,
            "args": ["-y", CONTEXT7_PACKAGE],
            "scope": "run-local",
            "env": mcp_env or {},
        },
    ]


def build_qwen_item_mcp_server_specs(
    *,
    registry_mode: str,
    project_dir: Path,
    attack_server_specs: list[dict[str, Any]],
    serena_command: Optional[str],
    serena_python: Optional[str],
    serena_context: str,
    context7_npx: Optional[str],
    mcp_env: Optional[dict[str, str]] = None,
) -> list[dict[str, Any]]:
    """Build the full MCP registry for one Qwen Code invocation."""
    specs = build_qwen_normal_mcp_server_specs(
        project_dir=project_dir,
        serena_command=serena_command,
        serena_python=serena_python,
        serena_context=serena_context,
        context7_npx=context7_npx,
        mcp_env=mcp_env,
    )
    if registry_mode == "mixed":
        specs.extend(attack_server_specs)
    return specs


def qwen_mcp_server_config(
    spec: dict[str, Any],
    *,
    timeout_ms: int,
    trust_mcp_servers: bool,
) -> dict[str, Any]:
    """Convert one internal MCP spec to Qwen Code settings.json format."""
    if spec["type"] == "http":
        config = {
            "httpUrl": spec["url"],
            "timeout": timeout_ms,
        }
    elif spec["type"] == "stdio":
        config = {
            "command": spec["command"],
            "args": spec.get("args", []),
            "timeout": timeout_ms,
        }
        if spec.get("cwd"):
            config["cwd"] = spec["cwd"]
        if spec.get("env"):
            config["env"] = spec["env"]
    else:
        raise ValueError(f"Unsupported MCP server type: {spec['type']}")

    if trust_mcp_servers:
        config["trust"] = True
    return config


def build_qwen_project_settings(
    *,
    model_id: str,
    openai_base_url: str,
    api_key_env: str,
    extra_body: dict[str, Any],
    server_specs: list[dict[str, Any]],
    timeout_ms: int,
    trust_mcp_servers: bool,
    skip_loop_detection: bool,
) -> dict[str, Any]:
    """Build a project-local Qwen settings payload without storing secrets."""
    server_names = [spec["name"] for spec in server_specs]
    return {
        "modelProviders": {
            "openai": [
                {
                    "id": model_id,
                    "name": f"OpenAI-compatible {model_id}",
                    "baseUrl": openai_base_url,
                    "envKey": api_key_env,
                    "generationConfig": {
                        "extra_body": extra_body,
                    }
                    if extra_body
                    else {},
                }
            ]
        },
        "security": {
            "auth": {
                "selectedType": "openai",
            }
        },
        "model": {
            "name": model_id,
            "skipLoopDetection": skip_loop_detection,
        },
        "mcp": {
            "allowed": server_names,
        },
        "mcpServers": {
            spec["name"]: qwen_mcp_server_config(
                spec,
                timeout_ms=timeout_ms,
                trust_mcp_servers=trust_mcp_servers,
            )
            for spec in server_specs
        },
        "telemetry": {
            "enabled": False,
            "logPrompts": False,
        },
        "general": {
            "checkpointing": {
                "enabled": False,
            },
            "enableAutoUpdate": False,
        },
    }


def write_qwen_project_files(
    *,
    workspace_dir: Path,
    trace_dir: Path,
    model_id: str,
    openai_base_url: str,
    api_key_env: str,
    extra_body: dict[str, Any],
    server_specs: list[dict[str, Any]],
    timeout_ms: int,
    trust_mcp_servers: bool,
    skip_loop_detection: bool,
) -> dict[str, str]:
    """Write project-local Qwen settings and trace-local user/system settings."""
    workspace_dir = workspace_dir.resolve()
    trace_dir = trace_dir.resolve()
    qwen_dir = workspace_dir / ".qwen"
    settings_path = qwen_dir / "settings.json"
    write_json_file(
        settings_path,
        build_qwen_project_settings(
            model_id=model_id,
            openai_base_url=openai_base_url,
            api_key_env=api_key_env,
            extra_body=extra_body,
            server_specs=server_specs,
            timeout_ms=timeout_ms,
            trust_mcp_servers=trust_mcp_servers,
            skip_loop_detection=skip_loop_detection,
        ),
    )

    # Stop Qwen Code from walking upward into repository .env files. Runtime
    # credentials are injected through the subprocess environment instead.
    write_text_file(
        workspace_dir / ".env",
        "# Intentionally blank. Qwen run credentials are injected by src.run_qwen_code.\n",
    )

    # These files are item-scoped Qwen settings, but the subprocess HOME must
    # stay run-scoped. If HOME points here, uv can create per-item .local trees.
    qwen_home = trace_dir / "qwen_home"
    qwen_home.mkdir(parents=True, exist_ok=True)
    system_defaults_path = qwen_home / "system-defaults.json"
    system_settings_path = qwen_home / "system-settings.json"
    write_json_file(system_defaults_path, {})
    write_json_file(system_settings_path, {})

    return {
        "qwen_settings_path": str(settings_path),
        "qwen_home": str(qwen_home),
        "qwen_system_defaults_path": str(system_defaults_path),
        "qwen_system_settings_path": str(system_settings_path),
    }


def redact_env_value(key: str, value: str) -> str:
    """Redact secrets in command trace files."""
    secret_keys = {
        "OPENAI_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
    }
    if key in secret_keys or key.endswith("_API_KEY") or key.endswith("_AUTH_TOKEN"):
        return "<set>" if value else "<empty>"
    return value


def build_isolated_qwen_env(extra_env: Optional[dict[str, str]]) -> dict[str, str]:
    """Build a subprocess env that avoids inherited Qwen/provider configuration."""
    blocked_prefixes = (
        "QWEN_",
        "GEMINI_",
        "OPENAI_",
        "ANTHROPIC_",
        "GOOGLE_",
        "TAVILY_",
    )
    base_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(blocked_prefixes)
    }
    return {**base_env, **(extra_env or {})}


def build_qwen_run_cache_paths(log_dir: Path) -> dict[str, Path]:
    """Return absolute cache dirs shared by all Qwen item workspaces in a run."""
    run_cache_root = (log_dir / "run_caches").resolve()
    return {
        "mcp_home": run_cache_root / "home",
        "xdg_config_home": run_cache_root / "xdg-config",
        "xdg_cache_home": run_cache_root / "xdg",
        "xdg_data_home": run_cache_root / "xdg-data",
        "npm_config_cache": run_cache_root / "npm",
        "uv_cache_dir": run_cache_root / "uv",
        "uv_tool_dir": run_cache_root / "uv-tools",
        "uv_tool_bin_dir": run_cache_root / "uv-tools-bin",
        "uv_python_install_dir": run_cache_root / "uv-python",
    }


def build_qwen_run_cache_env(cache_paths: dict[str, Path]) -> dict[str, str]:
    """Environment forcing uv/npm/MCP state into the current Qwen run directory."""
    return {
        "HOME": str(cache_paths["mcp_home"]),
        "XDG_CONFIG_HOME": str(cache_paths["xdg_config_home"]),
        "XDG_CACHE_HOME": str(cache_paths["xdg_cache_home"]),
        "XDG_DATA_HOME": str(cache_paths["xdg_data_home"]),
        "npm_config_cache": str(cache_paths["npm_config_cache"]),
        "NPM_CONFIG_CACHE": str(cache_paths["npm_config_cache"]),
        "UV_CACHE_DIR": str(cache_paths["uv_cache_dir"]),
        "UV_TOOL_DIR": str(cache_paths["uv_tool_dir"]),
        "UV_TOOL_BIN_DIR": str(cache_paths["uv_tool_bin_dir"]),
        "UV_PYTHON_INSTALL_DIR": str(cache_paths["uv_python_install_dir"]),
        "UV_NO_CONFIG": "1",
    }


def validate_qwen_run_cache_paths(
    cache_paths: dict[str, Path],
    *,
    workspace_dir: Path,
) -> None:
    """Fail closed if Qwen cache paths could be interpreted inside a workspace."""
    resolved_workspace = workspace_dir.resolve()
    for name, cache_path in cache_paths.items():
        resolved_cache_path = cache_path.resolve()
        if not resolved_cache_path.is_absolute():
            raise RuntimeError(f"{name} must be an absolute path: {cache_path}")
        try:
            resolved_cache_path.relative_to(resolved_workspace)
        except ValueError:
            continue
        raise RuntimeError(
            f"{name} must not be inside item workspace: {resolved_cache_path}"
        )


def run_qwen_code_for_item(
    item: QueryItem,
    *,
    qwen_binary: str,
    model_id: str,
    work_dir: str,
    timeout: int,
    system_prompt: str,
    output_format: str,
    include_partial_messages: bool,
    approval_mode: str,
    max_turns: int,
    max_tool_call_depth: Optional[int],
    registry_mode: str,
    extra_env: Optional[dict[str, str]] = None,
    trace_dir: Optional[str] = None,
) -> QwenCodeResult:
    """Run Qwen Code for one HumanEval item."""
    prompt = build_qwen_humaneval_prompt(
        item,
        registry_mode=registry_mode,
        system_prompt=system_prompt,
    )
    command = [
        qwen_binary,
        "--auth-type",
        "openai",
        "--model",
        model_id,
        "--output-format",
        output_format,
        "--approval-mode",
        approval_mode,
        "--max-session-turns",
        str(max_turns),
        "--chat-recording=false",
    ]
    if include_partial_messages and output_format == "stream-json":
        command.append("--include-partial-messages")
    command.append(prompt)

    step: dict[str, Any] = {
        "step_number": 1,
        "tool": "qwen_code",
        "command": command[:-1] + ["<prompt>"],
        "qwen_output_format": output_format,
        "registry_mode": registry_mode,
    }
    trace_path = Path(trace_dir) if trace_dir is not None else None
    if trace_path is not None:
        write_json_file(
            trace_path / "qwen_command.json",
            {
                "command": command[:-1] + ["<prompt>"],
                "cwd": work_dir,
                "env_overrides": {
                    key: redact_env_value(key, value)
                    for key, value in (extra_env or {}).items()
                },
            },
        )
        write_text_file(trace_path / "prompt.txt", prompt)

    try:
        returncode, stdout, stderr, runtime_error = run_agent_subprocess(
            command=command,
            work_dir=work_dir,
            timeout=timeout,
            env=build_isolated_qwen_env(extra_env),
            max_tool_call_depth=max_tool_call_depth if output_format == "stream-json" else None,
        )
    except subprocess.TimeoutExpired as exc:
        step["error"] = f"Timed out after {timeout}s"
        return QwenCodeResult(
            idx=item.idx,
            query=item.query,
            metadata=item.metadata,
            response=None,
            steps=[step],
            error=TimeoutError(str(exc)),
            trace_dir=str(trace_path) if trace_path else None,
            workspace_dir=work_dir,
        )

    step["returncode"] = returncode
    if stderr:
        step["stderr"] = stderr[-4000:]
    if runtime_error is not None:
        step["runtime_limit_error"] = runtime_error
    if trace_path is not None:
        write_text_file(trace_path / "stdout.txt", stdout)
        write_text_file(trace_path / "stderr.txt", stderr)

    raw_text, parsed_output = parse_qwen_output(stdout)
    parsed_metrics = parsed_output.get("metrics", {})
    metrics = compact_stream_metrics(parsed_metrics)
    result_event = parsed_metrics.get("result_event") if isinstance(parsed_metrics, dict) else {}
    qwen_reported_error = bool(
        isinstance(result_event, dict)
        and (
            result_event.get("is_error")
            or result_event.get("subtype") in {"error", "error_during_execution"}
        )
    )
    step["parsed_output_format"] = parsed_output.get("format")
    step["qwen_metrics"] = metrics
    step["token_usage"] = metrics["tokens"]
    step["tool_call_count"] = metrics["tool_call_count"]
    step["num_turns"] = metrics["num_turns"]
    step["api_retries"] = metrics["api_retries"]
    step["raw_response"] = raw_text
    if trace_path is not None:
        if parsed_output.get("format") in {"stream-json", "json"} and parsed_output.get("events"):
            write_jsonl_file(trace_path / "qwen_stream_events.jsonl", parsed_output.get("events", []))
            write_json_file(trace_path / "qwen_stream_metrics.json", parsed_metrics)
        elif parsed_output.get("payload") is not None:
            write_json_file(trace_path / "qwen_result_payload.json", parsed_output["payload"])

    if returncode != 0 or runtime_error is not None or qwen_reported_error:
        result_error = ""
        if isinstance(result_event, dict):
            error_payload = result_event.get("error")
            if isinstance(error_payload, dict):
                result_error = str(error_payload.get("message") or error_payload)
            elif error_payload:
                result_error = str(error_payload)
        message = (
            runtime_error
            or result_error
            or stderr.strip()
            or stdout.strip()
            or f"qwen exited {returncode}"
        )
        step["error"] = message[-4000:]
        step["final_status"] = "error"
        return QwenCodeResult(
            idx=item.idx,
            query=item.query,
            metadata=item.metadata,
            response=None,
            steps=[step],
            error=RuntimeError(message),
            trace_dir=str(trace_path) if trace_path else None,
            workspace_dir=work_dir,
        )

    completion = normalize_humaneval_completion(
        raw_text,
        prompt=item.metadata.get("prompt", item.query),
        entry_point=item.metadata.get("entry_point", ""),
    )
    step["normalized_response"] = completion
    step["final_status"] = "success"
    if trace_path is not None:
        write_text_file(trace_path / "raw_response.txt", raw_text)
        write_text_file(trace_path / "normalized_completion.py", completion)

    return QwenCodeResult(
        idx=item.idx,
        query=item.query,
        metadata=item.metadata,
        response=completion,
        steps=[step],
        error=None,
        trace_dir=str(trace_path) if trace_path else None,
        workspace_dir=work_dir,
    )


def summarize_result(result: QwenCodeResult) -> dict[str, Any]:
    """Build per-item trace summary from logger result and Qwen metrics."""
    step = result.steps[0] if result.steps else {}
    metrics = step.get("qwen_metrics") if isinstance(step.get("qwen_metrics"), dict) else {}
    usage = token_usage_from_mapping(metrics.get("tokens") or step.get("token_usage"))
    return {
        "idx": result.idx,
        "task_id": result.metadata.get("task_id"),
        "trace_dir": result.trace_dir,
        "workspace_dir": result.workspace_dir,
        "status": "error" if result.error else "success",
        "final_status": step.get("final_status") or ("error" if result.error else "success"),
        "error": str(result.error) if result.error else None,
        "num_turns": int(metrics.get("num_turns") or 0),
        "tokens": usage,
        "turns": metrics.get("turns", []),
        "tool_call_count": int(metrics.get("tool_call_count") or 0),
        "max_tool_call_depth_observed": int(metrics.get("max_tool_call_depth_observed") or 0),
        "available_tools": int(metrics.get("available_tools") or 0),
        "available_tool_names": metrics.get("available_tool_names", []),
        "mcp_servers": metrics.get("mcp_servers", []),
        "api_retries": int(metrics.get("api_retries") or 0),
        "duration_ms": metrics.get("duration_ms"),
        "duration_api_ms": metrics.get("duration_api_ms"),
        "total_cost_usd": float(metrics.get("total_cost_usd") or 0),
        "permission_denials": metrics.get("permission_denials", []),
        "model_usage": metrics.get("model_usage", {}),
    }


def aggregate_run_summary(*, results: list[QwenCodeResult]) -> dict[str, Any]:
    """Aggregate Qwen Code metrics for the run."""
    per_item = [summarize_result(result) for result in results]
    total_turns = sum(item["num_turns"] for item in per_item)
    total_tool_calls = sum(item["tool_call_count"] for item in per_item)
    total_tokens = {
        "input_tokens": sum(item["tokens"]["input_tokens"] for item in per_item),
        "output_tokens": sum(item["tokens"]["output_tokens"] for item in per_item),
        "total_tokens": sum(item["tokens"]["total_tokens"] for item in per_item),
    }
    per_turn = []
    for item in per_item:
        for turn in item["turns"]:
            per_turn.append(
                {
                    "idx": item["idx"],
                    "task_id": item["task_id"],
                    **turn,
                }
            )

    return {
        "qwen_code": {
            "total_items": len(results),
            "successful_items": sum(1 for result in results if result.error is None),
            "failed_items": sum(1 for result in results if result.error is not None),
            "total_turns": total_turns,
            "total_tool_calls": total_tool_calls,
            "max_tool_call_depth_observed": max(
                (item["max_tool_call_depth_observed"] for item in per_item),
                default=0,
            ),
            "total_available_tools": sum(item["available_tools"] for item in per_item),
            "available_tool_names_by_item": {
                str(item["idx"]): item["available_tool_names"] for item in per_item
            },
            "mcp_servers_by_item": {
                str(item["idx"]): item["mcp_servers"] for item in per_item
            },
            "total_api_retries": sum(item["api_retries"] for item in per_item),
            "total_cost_usd": sum(item["total_cost_usd"] for item in per_item),
            "total_duration_ms": sum(item["duration_ms"] or 0 for item in per_item),
            "total_duration_api_ms": sum(item["duration_api_ms"] or 0 for item in per_item),
            "tokens": total_tokens,
            "per_turn": per_turn,
            "per_item": per_item,
        }
    }


def classify_runtime_tool(tool_name: str) -> dict[str, Any]:
    """Classify a Qwen Code runtime tool name for tools.json."""
    if tool_name.startswith("mcp__"):
        parts = tool_name.split("__", 2)
        return {
            "name": tool_name,
            "source": "mcp",
            "mcp_server": parts[1] if len(parts) > 1 else None,
        }
    return {"name": tool_name, "source": "qwen_code_builtin"}


def build_runtime_tools_log(run_summary: dict[str, Any]) -> dict[str, Any]:
    """Build tools.json from Qwen Code stream-json init events."""
    qwen_summary = run_summary.get("qwen_code", {})
    tools_by_item = qwen_summary.get("available_tool_names_by_item", {})
    tools_by_name: dict[str, dict[str, Any]] = {}

    for item_idx, tool_names in tools_by_item.items():
        if not isinstance(tool_names, list):
            continue
        for tool_name in tool_names:
            if not isinstance(tool_name, str):
                continue
            tool_entry = tools_by_name.setdefault(
                tool_name,
                {
                    **classify_runtime_tool(tool_name),
                    "first_seen_item": item_idx,
                    "items": [],
                },
            )
            if item_idx not in tool_entry["items"]:
                tool_entry["items"].append(item_idx)

    tools = list(tools_by_name.values())
    return {
        "schema": "qwen_code_runtime_tools/v1",
        "capture_source": "qwen_code_stream_json_init",
        "total_unique_tools": len(tools),
        "tools": tools,
        "tools_by_item": tools_by_item,
        "mcp_servers_by_item": qwen_summary.get("mcp_servers_by_item", {}),
    }


def main(
    dataset_name: str = "human_eval",
    dataset_path: str = "dataset/human_eval.jsonl",
    model_id: Optional[str] = None,
    qwen_binary: str = "qwen",
    experiment_name: str = "qwen_code_humaneval",
    log_base_dir: str = "qwen_code_logs",
    num_workers: int = 1,
    timeout: int = 3600,
    limit: Optional[int] = None,
    log_dir_name: Optional[str] = None,
    workspace_mode: str = "empty",
    workspace_source: str = ".",
    base_url: Optional[str] = None,
    api_key_env: str = "OPENAI_API_KEY",
    registry_mode: str = "normal",
    mcp_python: Optional[str] = None,
    serena_command: Optional[str] = None,
    serena_python: Optional[str] = None,
    serena_context: str = "ide",
    context7_npx: Optional[str] = None,
    mcp_startup_timeout: int = 60,
    mcp_server_timeout_ms: int = QWEN_MCP_HTTP_TIMEOUT_MS,
    trust_mcp_servers: bool = True,
    skip_loop_detection: bool = False,
    max_turns: int = 1000,
    max_tool_call_depth: int = 20,
    approval_mode: str = "yolo",
    output_format: str = "stream-json",
    include_partial_messages: bool = True,
    unattended_retry: bool = False,
    nowait: bool = False,
    system_prompt: str = "You are a precise Python coding assistant. Output only the requested code.",
) -> None:
    """
    Run Qwen Code inference and HumanEval evaluation.

    Args:
        dataset_name: Name used by the evaluator. Keep "human_eval" for HumanEval.
        dataset_path: JSONL file containing OpenAI HumanEval rows.
        model_id: Model ID served by the endpoint. Defaults to OPENAI_MODEL_ID or VLLM_MODEL_ID.
        qwen_binary: Qwen Code executable.
        experiment_name: Name suffix for the log directory.
        log_base_dir: Directory where all Qwen Code logs/traces are stored.
        num_workers: Number of concurrent Qwen Code subprocesses.
        timeout: Per-problem subprocess timeout in seconds.
        limit: Optional sample cap for smoke tests. Default None runs the full dataset.
        log_dir_name: Optional existing/new log directory name.
        workspace_mode: "empty" creates one empty workspace per item; "copy" copies workspace_source.
        workspace_source: Source directory copied for each item when workspace_mode="copy".
        base_url: OpenAI-compatible endpoint. Defaults to OPENAI_BASE_URL or VLLM_BASE_URL.
        api_key_env: Environment variable holding the endpoint API key.
        registry_mode: "normal" enables Serena+Context7; "mixed" adds the cycle-inducing MCP tools.
        mcp_python: Python executable used for run-local FastMCP servers.
        serena_command: Legacy Serena command override. Must not point to uv/uvx.
        serena_python: Python executable whose virtualenv contains the Serena CLI.
        serena_context: Serena context for Qwen Code. "ide" is the generic coding-agent context.
        context7_npx: Optional npx executable override for Context7.
        mcp_startup_timeout: Seconds to wait for each run-local cycle-inducing MCP server port.
        mcp_server_timeout_ms: Timeout written into Qwen MCP server settings.
        trust_mcp_servers: Whether to mark project-local MCP servers trusted.
        skip_loop_detection: Disable Qwen Code loop detection.
        max_turns: Per-query Qwen Code turn cap set through --max-session-turns.
        max_tool_call_depth: Per-query runner-side hard cap for consecutive tool-use turns.
        approval_mode: Qwen Code approval mode. Use "yolo" for unattended MCP experiments.
        output_format: Qwen Code output format. Use stream-json for trace metrics.
        include_partial_messages: Capture partial stream messages when using stream-json.
        unattended_retry: Opt into Qwen Code's persistent retry mode for 429/529 responses.
        nowait: Enable NoWait mode by injecting logit_bias through Qwen OpenAI extra_body.
        system_prompt: Extra system prompt appended for the current Qwen Code run only.
    """
    backend = "qwen_code_openai_compatible"
    if workspace_mode not in {"empty", "copy"}:
        raise ValueError("workspace_mode must be 'empty' or 'copy'")
    if registry_mode not in REGISTRY_MODES:
        raise ValueError(f"registry_mode must be one of {sorted(REGISTRY_MODES)}")
    if output_format not in {"text", "json", "stream-json"}:
        raise ValueError("output_format must be 'text', 'json', or 'stream-json'")
    if approval_mode not in {"plan", "default", "auto-edit", "yolo"}:
        raise ValueError("approval_mode must be one of plan, default, auto-edit, yolo")
    include_partial_messages = coerce_bool(include_partial_messages, name="include_partial_messages")
    trust_mcp_servers = coerce_bool(trust_mcp_servers, name="trust_mcp_servers")
    skip_loop_detection = coerce_bool(skip_loop_detection, name="skip_loop_detection")
    unattended_retry = coerce_bool(unattended_retry, name="unattended_retry")
    nowait = coerce_bool(nowait, name="nowait")
    effective_skip_loop_detection = skip_loop_detection
    max_tool_call_depth_effective = max_tool_call_depth

    resolve_serena_executable(
        serena_command=serena_command,
        serena_python=serena_python,
    )
    resolve_required_executable("npx", context7_npx)

    resolved_model_id = model_id or os.getenv("OPENAI_MODEL_ID") or os.getenv("VLLM_MODEL_ID")
    if not resolved_model_id:
        raise ValueError("model_id is required via --model_id, OPENAI_MODEL_ID, or VLLM_MODEL_ID")

    resolved_openai_base_url = normalize_openai_base_url(resolve_qwen_base_url(base_url))
    resolved_binary = shutil.which(qwen_binary) or qwen_binary
    if not Path(resolved_binary).exists() and shutil.which(resolved_binary) is None:
        raise FileNotFoundError(f"Qwen Code binary not found: {qwen_binary}")
    openai_api_key = resolve_qwen_api_key(api_key_env)
    nowait_extra_body = build_nowait_extra_body(resolved_model_id, nowait)
    nowait_token_count = len(nowait_extra_body.get("logit_bias", {}))

    data_items = load_humaneval_items(dataset_path, limit)
    print(f"📊 Loaded {len(data_items)} samples from {dataset_path}")
    print(f"🤖 Backend: {backend}")
    print(f"🤖 Model: {resolved_model_id}")
    print(f"🔗 OpenAI-compatible base URL: {resolved_openai_base_url}")
    print(f"🚀 Workers: {num_workers}")
    print(f"🔧 Registry mode: {registry_mode}")
    if nowait:
        print(f"🚫 NoWait Mode: Enabled ({nowait_token_count} biased tokens)")

    evaluator = DatasetEvaluator(dataset_name=dataset_name)
    logger = ExperimentLogger(
        model_id=resolved_model_id,
        agent_type=backend,
        base_dir=log_base_dir,
        experiment_name=experiment_name,
        evaluator=evaluator,
        log_dir_name=log_dir_name,
        nowait=nowait,
    )
    logger.log_config(
        {
            "dataset_name": dataset_name,
            "dataset_path": dataset_path,
            "dataset_size": len(data_items),
            "backend": backend,
            "model_id": resolved_model_id,
            "qwen_binary": resolved_binary,
            "provider_base_url": resolved_openai_base_url,
            "api_key_env": api_key_env,
            "api_key_present": bool(openai_api_key),
            "registry_mode": registry_mode,
            "normal_mcp_servers": ["serena", "context7"],
            "attack_mcp_servers": list(ATTACK_MCP_SERVER_NAMES) if registry_mode == "mixed" else [],
            "mcp_python": resolve_mcp_python(mcp_python) if registry_mode == "mixed" else None,
            "serena_command": resolve_serena_executable(
                serena_command=serena_command,
                serena_python=serena_python,
            ),
            "serena_python": serena_python,
            "serena_context": serena_context,
            "context7_npx": resolve_required_executable("npx", context7_npx),
            "mcp_startup_timeout": mcp_startup_timeout,
            "mcp_server_timeout_ms": mcp_server_timeout_ms,
            "trust_mcp_servers": trust_mcp_servers,
            "skip_loop_detection_requested": skip_loop_detection,
            "skip_loop_detection_effective": effective_skip_loop_detection,
            "max_turns": max_turns,
            "max_tool_call_depth": max_tool_call_depth,
            "max_tool_call_depth_effective": max_tool_call_depth_effective,
            "approval_mode": approval_mode,
            "experiment_name": experiment_name,
            "log_base_dir": log_base_dir,
            "num_workers": num_workers,
            "timeout": timeout,
            "limit": limit,
            "workspace_mode": workspace_mode,
            "workspace_source": str(Path(workspace_source).resolve()),
            "output_format": output_format,
            "include_partial_messages": include_partial_messages,
            "unattended_retry": unattended_retry,
            "nowait": nowait,
            "nowait_token_count": nowait_token_count,
            "global_qwen_config_touched": False,
        }
    )
    run_log_dir = logger.log_dir.resolve()
    trace_root = run_log_dir / "item_traces"
    workspace_root = run_log_dir / "workspaces"
    run_cache_paths = build_qwen_run_cache_paths(run_log_dir)
    for cache_path in run_cache_paths.values():
        cache_path.mkdir(parents=True, exist_ok=True)
    mcp_env = build_qwen_run_cache_env(run_cache_paths)
    write_json_file(
        run_log_dir / "run_manifest.json",
        {
            "created_at": datetime.now().isoformat(),
            "backend": backend,
            "model_id": resolved_model_id,
            "dataset_path": dataset_path,
            "dataset_size": len(data_items),
            "num_workers": num_workers,
            "output_format": output_format,
            "include_partial_messages": include_partial_messages,
            "registry_mode": registry_mode,
            "max_turns": max_turns,
            "max_tool_call_depth": max_tool_call_depth,
            "max_tool_call_depth_effective": max_tool_call_depth_effective,
            "skip_loop_detection_effective": effective_skip_loop_detection,
            "nowait": nowait,
            "nowait_token_count": nowait_token_count,
            "trace_root": str(trace_root),
            "workspace_root": str(workspace_root),
            "run_cache_paths": {
                key: str(path)
                for key, path in run_cache_paths.items()
            },
        },
    )

    attack_mcp_server_specs: list[dict[str, Any]] = []
    mcp_processes: list[subprocess.Popen] = []
    if registry_mode == "mixed":
        attack_mcp_server_specs, mcp_processes = start_run_local_mcp_servers(
            log_dir=run_log_dir,
            mcp_python=mcp_python,
            startup_timeout=mcp_startup_timeout,
            cache_env=mcp_env,
        )
        print(f"🔧 Started {len(attack_mcp_server_specs)} run-local cycle-inducing MCP servers")
    write_json_file(
        run_log_dir / "mcp_servers.json",
        {
            "registry_mode": registry_mode,
            "normal_mcp_servers": build_qwen_normal_mcp_server_specs(
                project_dir="${ITEM_WORKSPACE}",
                serena_command=serena_command,
                serena_python=serena_python,
                serena_context=serena_context,
                context7_npx=context7_npx,
                mcp_env=mcp_env,
            ),
            "attack_mcp_servers": attack_mcp_server_specs,
            "qwen_settings_scope": "item-workspace/.qwen/settings.json",
            "global_qwen_config_touched": False,
        },
    )

    worker_kwargs = {
        "qwen_binary": resolved_binary,
        "model_id": resolved_model_id,
        "timeout": timeout,
        "system_prompt": system_prompt,
        "output_format": output_format,
        "include_partial_messages": include_partial_messages,
        "approval_mode": approval_mode,
        "max_turns": max_turns,
        "max_tool_call_depth": max_tool_call_depth_effective,
        "registry_mode": registry_mode,
    }

    def run_one_item(item: QueryItem) -> QwenCodeResult:
        workspace_dir = prepare_item_workspace(
            item,
            workspace_root=workspace_root,
            workspace_mode=workspace_mode,
            workspace_source=Path(workspace_source).resolve(),
        )
        validate_qwen_run_cache_paths(
            run_cache_paths,
            workspace_dir=workspace_dir,
        )
        trace_dir = trace_root / item_slug(item)
        trace_dir.mkdir(parents=True, exist_ok=True)
        item_mcp_server_specs = build_qwen_item_mcp_server_specs(
            registry_mode=registry_mode,
            project_dir=workspace_dir.resolve(),
            attack_server_specs=attack_mcp_server_specs,
            serena_command=serena_command,
            serena_python=serena_python,
            serena_context=serena_context,
            context7_npx=context7_npx,
            mcp_env=mcp_env,
        )
        qwen_paths = write_qwen_project_files(
            workspace_dir=workspace_dir,
            trace_dir=trace_dir,
            model_id=resolved_model_id,
            openai_base_url=resolved_openai_base_url,
            api_key_env=QWEN_PROVIDER_API_KEY_ENV,
            extra_body=nowait_extra_body,
            server_specs=item_mcp_server_specs,
            timeout_ms=mcp_server_timeout_ms,
            trust_mcp_servers=trust_mcp_servers,
            skip_loop_detection=effective_skip_loop_detection,
        )
        write_json_file(trace_dir / "mcp_servers.json", item_mcp_server_specs)
        write_json_file(trace_dir / "qwen_runtime_paths.json", qwen_paths)

        qwen_env = {
            **mcp_env,
            "QWEN_CODE_SYSTEM_DEFAULTS_PATH": qwen_paths["qwen_system_defaults_path"],
            "QWEN_CODE_SYSTEM_SETTINGS_PATH": qwen_paths["qwen_system_settings_path"],
            "QWEN_CODE_NO_RELAUNCH": "true",
            "QWEN_DEFAULT_AUTH_TYPE": "openai",
            "OPENAI_API_KEY": openai_api_key,
            "OPENAI_BASE_URL": resolved_openai_base_url,
            "OPENAI_MODEL": resolved_model_id,
            "NO_COLOR": "1",
        }
        if unattended_retry:
            qwen_env["QWEN_CODE_UNATTENDED_RETRY"] = "1"

        item_worker_kwargs = dict(worker_kwargs)
        item_worker_kwargs["trace_dir"] = str(trace_dir)
        item_worker_kwargs["work_dir"] = str(workspace_dir)
        item_worker_kwargs["extra_env"] = qwen_env

        result = run_qwen_code_for_item(item, **item_worker_kwargs)
        result.workspace_dir = str(workspace_dir)
        result.trace_dir = str(trace_dir)
        result.metadata = {
            **result.metadata,
            "workspace_dir": str(workspace_dir),
            "trace_dir": str(trace_dir),
            **qwen_paths,
        }
        write_json_file(trace_dir / "item_summary.json", summarize_result(result))
        return result

    try:
        if num_workers <= 1:
            results = [
                run_one_item(item)
                for item in tqdm(data_items, desc="Processing samples")
            ]
        else:
            results = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(run_one_item, item): item
                    for item in data_items
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
                    results.append(future.result())
    finally:
        stop_run_local_mcp_servers(mcp_processes)

    results.sort(key=lambda result: result.idx)
    for result in results:
        query_idx = logger.start_query(result.query, metadata=result.metadata)
        logger.log_steps(query_idx, result.steps)
        if result.error is not None:
            logger.log_error(query_idx, result.error)
        else:
            logger.end_query(query_idx, result.response or "")

    run_summary = aggregate_run_summary(results=results)
    runtime_tools_log = build_runtime_tools_log(run_summary)
    write_json_file(run_log_dir / "tools.json", runtime_tools_log)
    logger.metadata["tools"] = runtime_tools_log["tools"]
    logger.metadata["num_tools"] = runtime_tools_log["total_unique_tools"]
    logger.metadata["tool_capture_source"] = runtime_tools_log["capture_source"]

    logger.finalize()
    write_json_file(run_log_dir / "qwen_code_run_summary.json", run_summary)
    print(f"\n✅ {backend} HumanEval run complete: {run_log_dir}")


if __name__ == "__main__":
    load_dotenv(override=True)
    fire.Fire(main)
