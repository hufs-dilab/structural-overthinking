"""
OpenAI model wrapper with pluggable tool call parsing.
"""

import uuid
from typing import List, Optional

from smolagents import OpenAIModel
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole

from src.parsers import ToolCallParser


class ParsedOpenAIModel(OpenAIModel):
    """
    OpenAI model with custom tool call parser support.

    When the API doesn't return structured tool_calls, this class
    uses a pluggable parser to extract them from the content.
    """

    def __init__(self, *args, parser: Optional[ToolCallParser] = None, **kwargs):
        """
        Initialize model with optional custom parser.

        Args:
            parser: Custom tool call parser instance
            *args, **kwargs: Passed to OpenAIModel
        """
        super().__init__(*args, **kwargs)
        self._parser = parser

    @property
    def parser(self) -> Optional[ToolCallParser]:
        return self._parser

    @parser.setter
    def parser(self, value: Optional[ToolCallParser]) -> None:
        self._parser = value

    def parse_tool_calls(self, message: ChatMessage) -> ChatMessage:
        """
        Parse tool calls using custom parser if available.

        Falls back to parent implementation if parser doesn't match
        or if tool_calls are already present with valid arguments.
        """
        message.role = MessageRole.ASSISTANT

        # Check if tool_calls exist and have valid arguments
        if message.tool_calls and len(message.tool_calls) > 0:
            # Check if any tool call has empty arguments - if so, try content parsing
            has_empty_args = any(
                not tc.function.arguments or tc.function.arguments == "{}" for tc in message.tool_calls
            )
            if not has_empty_args:
                return super().parse_tool_calls(message)
            # If tool_calls have empty args, try to parse from content

        # Try custom parser if available and applicable
        if self._parser and message.content:
            if self._parser.can_parse(message.content):
                parsed = self._parser.parse(message.content)
                if parsed:
                    message.tool_calls = [
                        ChatMessageToolCall(
                            id=call.id or str(uuid.uuid4()),
                            type="function",
                            function=ChatMessageToolCallFunction(
                                name=call.name,
                                arguments=call.arguments,
                            ),
                        )
                        for call in parsed
                    ]
                    return message

        # Fallback to parent implementation
        return super().parse_tool_calls(message)