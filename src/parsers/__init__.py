"""
Tool call parsers for various model formats.

This module provides a registry-based system for parsing tool calls
from different model output formats (JSON, XML, etc.).
"""

from .base import ParsedToolCall, ToolCallParser
from .deepseek_v3 import DeepSeekV3Parser, extract_deepseek_tool_calls
from .kimi_k2 import KimiK2Parser, extract_tool_call_info
from .registry import get_parser, list_parsers, register_parser

__all__ = [
    "ToolCallParser",
    "ParsedToolCall",
    "get_parser",
    "register_parser",
    "list_parsers",
    "DeepSeekV3Parser",
    "extract_deepseek_tool_calls",
    "KimiK2Parser",
    "extract_tool_call_info",
]