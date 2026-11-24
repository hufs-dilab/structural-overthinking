"""
DeepSeek V3.2 tool call parser.

DeepSeek V3.2 uses special tokens for tool calls when the server doesn't provide
native tool-call parsing:
- <｜DSML｜function_calls> ... </｜DSML｜function_calls>
- <｜DSML｜invoke name="func_name"> ... </｜DSML｜invoke>
- <｜DSML｜parameter name="param_name" string="true">value</｜DSML｜parameter>

This parser handles the fallback format when the vLLM server is not configured with
--tool-call-parser deepseek_v32.

Reference: https://docs.vllm.ai/en/latest/features/tool_calling.html
"""

import json
import re
from typing import Any, List

from .base import ParsedToolCall, ToolCallParser


class DeepSeekV3Parser(ToolCallParser):
    """Parser for DeepSeek V3.2 special token format tool calls."""

    name = "deepseek_v3"
    # Auto-detect for deepseek-v3 models
    model_patterns = ["deepseek-v3", "deepseek_v3"]

    # Pattern to extract function calls section
    # Using both full-width and ASCII variants of the special tokens
    _FUNCTION_CALLS_PATTERN = re.compile(
        r"<[｜|]DSML[｜|]function_calls>(.*?)</[｜|]DSML[｜|]function_calls>", re.DOTALL
    )

    # Pattern to extract individual invoke blocks
    _INVOKE_PATTERN = re.compile(
        r'<[｜|]DSML[｜|]invoke\s+name="([^"]+)"[^>]*>(.*?)</[｜|]DSML[｜|]invoke>', re.DOTALL
    )

    # Pattern to extract parameters
    # Allow any character in parameter value including <
    _PARAMETER_PATTERN = re.compile(
        r'<[｜|]DSML[｜|]parameter\s+name="([^"]+)"[^>]*>(.*?)</[｜|]DSML[｜|]parameter>', re.DOTALL
    )

    def can_parse(self, content: str) -> bool:
        """Check if content contains DeepSeek V3 tool call markers."""
        if not content:
            return False
        # Handle both direct content and content wrapped in [{'type': 'text', ...}] format
        return "<｜DSML｜function_calls>" in content or "<|DSML|function_calls>" in content

    def _normalize_content(self, content: str) -> str:
        """Normalize content that may be wrapped in [{'type': 'text', ...}] format."""
        # Check if content looks like a JSON-like list of dicts (smolagents format)
        if content.startswith("[{") and "'type': 'text'" in content:
            # Try to extract the actual text content
            import ast

            try:
                # Handle single-quoted JSON-like format
                parsed = ast.literal_eval(content)
                if isinstance(parsed, list) and parsed:
                    texts = []
                    for item in parsed:
                        if isinstance(item, dict) and item.get("type") == "text":
                            texts.append(item.get("text", ""))
                    return "\n".join(texts)
            except (ValueError, SyntaxError):
                pass
        return content

    def parse(self, content: str) -> List[ParsedToolCall]:
        """Parse DeepSeek V3 special token format tool calls."""
        if not self.can_parse(content):
            return []

        # Normalize content if it's in wrapped format
        content = self._normalize_content(content)

        tool_calls = []

        try:
            # Extract function calls section
            section_match = self._FUNCTION_CALLS_PATTERN.search(content)
            if not section_match:
                return []

            function_calls_section = section_match.group(1)

            # Extract individual invoke blocks
            for invoke_match in self._INVOKE_PATTERN.finditer(function_calls_section):
                function_name = invoke_match.group(1)
                invoke_content = invoke_match.group(2)

                # Extract parameters
                arguments = {}
                for param_match in self._PARAMETER_PATTERN.finditer(invoke_content):
                    param_name = param_match.group(1)
                    param_value = param_match.group(2).strip()

                    # Try to parse as JSON first (for complex types)
                    try:
                        arguments[param_name] = json.loads(param_value)
                    except json.JSONDecodeError:
                        # Keep as string
                        arguments[param_name] = param_value

                tool_calls.append(
                    ParsedToolCall(
                        name=function_name,
                        arguments=arguments,
                    )
                )

        except Exception as e:
            print(f"[DeepSeekV3Parser] Failed to parse: {e}")

        return tool_calls


def extract_deepseek_tool_calls(content: str) -> List[dict]:
    """
    Extract tool calls from DeepSeek V3 raw output.

    Standalone function for easy use.

    Args:
        content: Raw model output containing tool call markers

    Returns:
        List of tool call dictionaries with 'name' and 'arguments' keys
    """
    parser = DeepSeekV3Parser()
    parsed = parser.parse(content)
    return [{"name": tc.name, "arguments": tc.arguments} for tc in parsed]
