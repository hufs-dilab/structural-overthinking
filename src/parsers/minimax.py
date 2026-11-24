"""
MiniMax XML format tool call parser.

MiniMax-M2/M2.1 can use a custom XML format for tool calls when using native tool calling:
<minimax:tool_call>
<invoke name="function_name">
<parameter name="param_name">param_value</parameter>
</invoke>
</minimax:tool_call>

NOTE: When using smolagents with MiniMax, the model typically outputs JSON format
with <think> tags instead. Use 'thinking_json' parser for that case.

Reference: https://huggingface.co/MiniMaxAI/MiniMax-M2.1/blob/main/docs/tool_calling_guide.md
"""

import json
import re
from typing import Any, List

from .base import ParsedToolCall, ToolCallParser


def _extract_name(name_str: str) -> str:
    """Extract name from quoted string."""
    name_str = name_str.strip()
    if (name_str.startswith('"') and name_str.endswith('"')) or (name_str.startswith("'") and name_str.endswith("'")):
        return name_str[1:-1]
    return name_str


def _convert_value(value: str, param_type: str) -> Any:
    """Convert parameter value based on type."""
    if value.lower() == "null":
        return None

    param_type = param_type.lower()

    if param_type in ("string", "str", "text"):
        return value
    elif param_type in ("integer", "int"):
        try:
            return int(value)
        except (ValueError, TypeError):
            return value
    elif param_type in ("number", "float"):
        try:
            val = float(value)
            return int(val) if val == int(val) else val
        except (ValueError, TypeError):
            return value
    elif param_type in ("boolean", "bool"):
        return value.lower() in ("true", "1")
    elif param_type in ("object", "array"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    else:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value


class MiniMaxParser(ToolCallParser):
    """Parser for MiniMax-M2/M2.1 XML tool call format."""

    name = "minimax_xml"
    # Don't auto-detect - XML format is only used with native tool calling, not smolagents
    model_patterns = []

    # Compiled regex patterns
    _TOOL_CALL_PATTERN = re.compile(r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL)
    _INVOKE_PATTERN = re.compile(r"<invoke name=(.*?)</invoke>", re.DOTALL)
    _PARAM_PATTERN = re.compile(r"<parameter name=(.*?)</parameter>", re.DOTALL)

    def can_parse(self, content: str) -> bool:
        """Check if content contains MiniMax tool call markers."""
        return "<minimax:tool_call>" in content

    def parse(self, content: str) -> List[ParsedToolCall]:
        """Parse MiniMax XML format tool calls."""
        if not self.can_parse(content):
            return []

        tool_calls = []

        try:
            for tool_call_match in self._TOOL_CALL_PATTERN.findall(content):
                for invoke_match in self._INVOKE_PATTERN.findall(tool_call_match):
                    # Extract function name
                    name_match = re.search(r"^([^>]+)", invoke_match)
                    if not name_match:
                        continue

                    function_name = _extract_name(name_match.group(1))
                    param_config = self.get_param_config(function_name)

                    # Extract parameters
                    arguments = {}
                    for match in self._PARAM_PATTERN.findall(invoke_match):
                        param_match = re.search(r"^([^>]+)>(.*)", match, re.DOTALL)
                        if param_match:
                            param_name = _extract_name(param_match.group(1))
                            param_value = param_match.group(2).strip()

                            # Clean up newlines
                            param_value = param_value.strip("\n")

                            # Get type and convert
                            param_type = "string"
                            if param_name in param_config:
                                type_info = param_config[param_name]
                                if isinstance(type_info, dict) and "type" in type_info:
                                    param_type = type_info["type"]

                            arguments[param_name] = _convert_value(param_value, param_type)

                    tool_calls.append(ParsedToolCall(name=function_name, arguments=arguments))

        except Exception as e:
            print(f"[MiniMaxParser] Failed to parse tool calls: {e}")
            return []

        return tool_calls