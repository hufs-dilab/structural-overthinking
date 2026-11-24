"""
Kimi K2 tool call parser.

Kimi K2 uses special tokens for tool calls when manual parsing is needed:
- <|tool_calls_section_begin|> ... <|tool_calls_section_end|>
- <|tool_call_begin|> tool_id <|tool_call_argument_begin|> arguments <|tool_call_end|>

Tool ID format: functions.{func_name}:{idx}

This parser is for cases where the server doesn't provide a tool-call parser
and returns raw output with special tokens in the content field.

Reference: https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/main/docs/tool_calling_guide.md
"""

import json
import re
from typing import List

from .base import ParsedToolCall, ToolCallParser


class KimiK2Parser(ToolCallParser):
    """Parser for Kimi K2 special token format tool calls."""

    name = "kimi_k2"
    # Don't auto-detect - only use when explicitly specified or when special tokens are found
    # The API normally returns structured tool_calls, so this parser is only needed
    # when the server doesn't provide a tool-call parser
    model_patterns = []

    # Pattern to extract tool calls section
    _TOOL_CALLS_SECTION_PATTERN = re.compile(
        r"<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>", re.DOTALL
    )

    # Pattern to extract individual tool calls
    _TOOL_CALL_PATTERN = re.compile(
        r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*"
        r"<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*?)\s*"
        r"<\|tool_call_end\|>",
        re.DOTALL,
    )

    def can_parse(self, content: str) -> bool:
        """Check if content contains Kimi K2 tool call markers."""
        return "<|tool_calls_section_begin|>" in content

    def parse(self, content: str) -> List[ParsedToolCall]:
        """Parse Kimi K2 special token format tool calls."""
        if not self.can_parse(content):
            return []

        tool_calls = []

        try:
            # Extract tool calls section
            section_match = self._TOOL_CALLS_SECTION_PATTERN.search(content)
            if not section_match:
                return []

            tool_calls_section = section_match.group(1)

            # Extract individual tool calls
            for match in self._TOOL_CALL_PATTERN.finditer(tool_calls_section):
                function_id = match.group("tool_call_id")
                function_args_str = match.group("function_arguments").strip()

                # Parse function name from ID: functions.{func_name}:{idx}
                # Example: functions.get_weather:0 -> get_weather
                try:
                    function_name = function_id.split(".")[1].split(":")[0]
                except (IndexError, AttributeError):
                    # Fallback: use the full ID
                    function_name = function_id

                # Parse arguments as JSON
                try:
                    arguments = json.loads(function_args_str)
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to extract key-value pairs
                    arguments = self._parse_malformed_args(function_args_str)

                tool_calls.append(ParsedToolCall(name=function_name, arguments=arguments, id=function_id))

        except Exception as e:
            print(f"[KimiK2Parser] Failed to parse: {e}")

        return tool_calls


    def _parse_malformed_args(self, args_str: str) -> dict:
        """Try to extract arguments from malformed JSON."""
        args = {}

        # Try to find key-value pairs
        # Match "key": "value" or "key": number or "key": true/false
        kv_pattern = r'"([^"]+)"\s*:\s*(?:"([^"\\]*(?:\\.[^"\\]*)*)"|(-?\d+(?:\.\d+)?)|(\btrue\b|\bfalse\b|\bnull\b))'

        for match in re.finditer(kv_pattern, args_str, re.IGNORECASE):
            key = match.group(1)
            if match.group(2) is not None:  # string value
                try:
                    value = match.group(2).encode().decode("unicode_escape")
                except:
                    value = match.group(2)
                args[key] = value
            elif match.group(3) is not None:  # number
                num_str = match.group(3)
                args[key] = float(num_str) if "." in num_str else int(num_str)
            elif match.group(4) is not None:  # boolean or null
                val = match.group(4).lower()
                if val == "true":
                    args[key] = True
                elif val == "false":
                    args[key] = False
                else:
                    args[key] = None

        return args


def extract_tool_call_info(tool_call_rsp: str) -> List[dict]:
    """
    Extract tool call info from Kimi K2 raw output.

    This is a standalone function matching the reference implementation
    from the Kimi K2 documentation.

    Args:
        tool_call_rsp: Raw model output containing tool call markers

    Returns:
        List of tool call dictionaries with 'id', 'type', and 'function' keys
    """
    if "<|tool_calls_section_begin|>" not in tool_call_rsp:
        return []

    pattern = r"<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>"
    tool_calls_sections = re.findall(pattern, tool_call_rsp, re.DOTALL)

    if not tool_calls_sections:
        return []

    func_call_pattern = (
        r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*"
        r"<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*?)\s*"
        r"<\|tool_call_end\|>"
    )

    tool_calls = []
    for match in re.findall(func_call_pattern, tool_calls_sections[0], re.DOTALL):
        function_id, function_args = match
        # function_id: functions.get_weather:0
        function_name = function_id.split(".")[1].split(":")[0]
        tool_calls.append(
            {"id": function_id, "type": "function", "function": {"name": function_name, "arguments": function_args}}
        )

    return tool_calls