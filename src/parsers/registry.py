"""
Parser registry for tool call formats.

Provides automatic parser selection based on model ID patterns,
with support for explicit parser specification.
"""

from typing import Dict, List, Optional, Type

from .base import ToolCallParser

# Global registry
_PARSERS: Dict[str, Type[ToolCallParser]] = {}


def register_parser(parser_class: Type[ToolCallParser]) -> Type[ToolCallParser]:
    """
    Register a parser class.

    Can be used as a decorator:
        @register_parser
        class MyParser(ToolCallParser):
            ...

    Args:
        parser_class: Parser class to register

    Returns:
        The same parser class (for decorator usage)
    """
    _PARSERS[parser_class.name] = parser_class
    return parser_class


def get_parser(
    name: Optional[str] = None,
    model_id: Optional[str] = None,
    tools_schema: Optional[List] = None,
) -> Optional[ToolCallParser]:
    """
    Get a parser instance.

    Args:
        name: Explicit parser name (takes precedence)
        model_id: Model ID for auto-detection
        tools_schema: Tools schema for type conversion

    Returns:
        Parser instance, or None if no match found
    """
    parser_class = None

    # Explicit name takes precedence
    if name:
        parser_class = _PARSERS.get(name)
        if not parser_class:
            available = list(_PARSERS.keys())
            raise ValueError(f"Unknown parser: {name}. Available: {available}")

    # Auto-detect from model ID
    elif model_id:
        for cls in _PARSERS.values():
            if cls.matches_model(model_id):
                parser_class = cls
                break

    if parser_class:
        return parser_class(tools_schema=tools_schema)

    return None


def list_parsers() -> List[str]:
    """List all registered parser names."""
    return list(_PARSERS.keys())


# Auto-register built-in parsers
def _register_builtin_parsers():
    """Register built-in parsers."""
    from .deepseek_v3 import DeepSeekV3Parser
    from .kimi_k2 import KimiK2Parser
    from .minimax import MiniMaxParser
    from .thinking_json import ThinkingJsonParser

    register_parser(DeepSeekV3Parser)
    register_parser(KimiK2Parser)
    register_parser(MiniMaxParser)
    register_parser(ThinkingJsonParser)


_register_builtin_parsers()