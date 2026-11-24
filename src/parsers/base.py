"""
Base classes and protocols for tool call parsers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ParsedToolCall:
    """Represents a parsed tool call."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None


class ToolCallParser(ABC):
    """
    Abstract base class for tool call parsers.

    Implement this to support custom tool call formats for different models.
    """

    # Unique identifier for this parser
    name: str = "base"

    # Model ID patterns this parser handles (for auto-detection)
    # Override in subclasses, e.g., ["minimax", "m2"]
    model_patterns: List[str] = []

    def __init__(self, tools_schema: Optional[List[Dict]] = None):
        """
        Initialize parser with optional tools schema for type conversion.

        Args:
            tools_schema: List of tool definitions with parameter types
        """
        self.tools_schema = tools_schema or []

    def set_tools_schema(self, tools: List) -> None:
        """
        Set tools schema from smolagents Tool objects.

        Args:
            tools: List of Tool objects or dict definitions
        """
        self.tools_schema = []
        for tool in tools:
            if hasattr(tool, "name") and hasattr(tool, "inputs"):
                self.tools_schema.append(
                    {
                        "name": tool.name,
                        "parameters": {"type": "object", "properties": tool.inputs},
                    }
                )
            elif isinstance(tool, dict):
                self.tools_schema.append(tool)

    def get_param_config(self, function_name: str) -> Dict[str, Any]:
        """Get parameter configuration for a function."""
        for tool in self.tools_schema:
            tool_name = tool.get("name") or tool.get("function", {}).get("name")
            if tool_name == function_name:
                params = tool.get("parameters") or tool.get("function", {}).get("parameters")
                if isinstance(params, dict) and "properties" in params:
                    return params["properties"]
        return {}

    @abstractmethod
    def can_parse(self, content: str) -> bool:
        """
        Check if this parser can handle the given content.

        Args:
            content: Model output text

        Returns:
            True if this parser can parse the content
        """
        pass

    @abstractmethod
    def parse(self, content: str) -> List[ParsedToolCall]:
        """
        Parse tool calls from model output.

        Args:
            content: Model output text

        Returns:
            List of parsed tool calls
        """
        pass

    @classmethod
    def matches_model(cls, model_id: str) -> bool:
        """
        Check if this parser should be used for the given model.

        Args:
            model_id: Model identifier

        Returns:
            True if this parser matches the model
        """
        model_id_lower = model_id.lower()
        return any(pattern in model_id_lower for pattern in cls.model_patterns)
