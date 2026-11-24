"""
Thinking model JSON tool call parser.

Handles models that output tool calls in JSON format with <think> tags:
<think>reasoning...</think>[{"name": "tool", "parameters": {...}}]

This is common in reasoning models like DeepSeek R1, Kimi K2, Qwen-thinking, MiniMax-M2, etc.
"""

import json
import re
from typing import List

from .base import ParsedToolCall, ToolCallParser


class ThinkingJsonParser(ToolCallParser):
    """Parser for thinking models that output JSON tool calls with <think> tags."""

    name = "thinking_json"
    # Note: Models like kimi-k2 and qwen3-thinking are excluded because their APIs
    # already return structured tool_calls natively. This parser is only for models
    # that output tool calls as JSON text in content.
    model_patterns = [
        "deepseek-r1",
        "minimax",
    ]

    # Pattern to extract content after </think> tag
    _THINK_PATTERN = re.compile(r"</think>\s*", re.DOTALL)
    # Pattern to find JSON array of tool calls (handles escaped brackets like \])
    _JSON_ARRAY_PATTERN = re.compile(r"\[.*?\](?:\\?\])?", re.DOTALL)

    def can_parse(self, content: str) -> bool:
        """Check if content has think tags or JSON tool call format."""
        if "<think>" in content or "</think>" in content:
            return True
        # Also try to detect JSON array format without think tags
        if content.strip().startswith("[{") and '"name"' in content:
            return True
        return False

    def parse(self, content: str) -> List[ParsedToolCall]:
        """Parse tool calls from thinking model output."""
        if not content:
            return []

        # Remove <think>...</think> content
        # First try to find content after </think>
        if "</think>" in content:
            parts = self._THINK_PATTERN.split(content)
            if len(parts) > 1:
                content = parts[-1].strip()

        # Also remove any remaining <think> tags (in case of malformed output)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        content = re.sub(r"<think>.*", "", content, flags=re.DOTALL)
        content = content.strip()

        if not content:
            return []

        tool_calls = []

        try:
            # Clean up common issues:
            # 1. Remove escaped brackets at the end (e.g., \])
            content = re.sub(r"\\?\]$", "]", content)
            # 2. Fix common JSON formatting issues
            content = content.replace("\\]", "]")

            # Try to find and parse JSON array
            # Look for [{ ... }] pattern
            json_match = re.search(r"\[\s*\{.*\}\s*\]", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)

                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "name" in item:
                            name = item["name"]
                            # Handle both "parameters" and "arguments" keys
                            arguments = item.get("parameters", item.get("arguments", {}))

                            # If arguments is a string, try to parse it as JSON
                            if isinstance(arguments, str):
                                try:
                                    arguments = json.loads(arguments)
                                except json.JSONDecodeError:
                                    pass

                            tool_calls.append(ParsedToolCall(name=name, arguments=arguments))

        except json.JSONDecodeError as e:
            print(f"[ThinkingJsonParser] JSON parse error: {e}")
            # Try to extract from malformed JSON
            tool_calls = self._parse_malformed(content)

        except Exception as e:
            print(f"[ThinkingJsonParser] Failed to parse: {e}")

        return tool_calls

    def _parse_malformed(self, content: str) -> List[ParsedToolCall]:
        """Try to extract tool calls from malformed JSON."""
        tool_calls = []

        # Try to find individual tool call objects with nested JSON in arguments
        # Match: {"name": "xxx", "parameters": {"key": "value..."}}
        pattern = r'\{\s*"name"\s*:\s*"([^"]+)"[^}]*"(?:parameters|arguments)"\s*:\s*(\{(?:[^{}]|\{[^{}]*\})*\})'
        matches = re.findall(pattern, content, re.DOTALL)

        for name, args_str in matches:
            try:
                arguments = json.loads(args_str)
                tool_calls.append(ParsedToolCall(name=name, arguments=arguments))
            except json.JSONDecodeError:
                # Try extracting key-value pairs with regex
                args = {}
                # Match "key": "value" or "key": number
                kv_pattern = r'"([^"]+)"\s*:\s*(?:"([^"\\]*(?:\\.[^"\\]*)*)"|(\d+(?:\.\d+)?)|(\btrue\b|\bfalse\b))'
                for match in re.finditer(kv_pattern, args_str):
                    key = match.group(1)
                    if match.group(2) is not None:  # string value
                        value = match.group(2).encode().decode("unicode_escape")
                        args[key] = value
                    elif match.group(3) is not None:  # number
                        num_str = match.group(3)
                        args[key] = float(num_str) if "." in num_str else int(num_str)
                    elif match.group(4) is not None:  # boolean
                        args[key] = match.group(4).lower() == "true"

                if args:
                    tool_calls.append(ParsedToolCall(name=name, arguments=args))
                else:
                    tool_calls.append(ParsedToolCall(name=name, arguments={}))

        return tool_calls
