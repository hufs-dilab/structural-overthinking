"""
Tool loading and management utilities.
"""

import json
import random
from collections import defaultdict
from collections.abc import Callable
from typing import Dict, List

from smolagents import Tool

ToolRetExecutor = Callable[[str, str], object]


def load_mcp_tools() -> List[Tool]:
    """
    Load tools from the cycle-inducing MCP servers in tools/.

    Returns:
        List of Tool instances from MCP servers
    """
    import asyncio
    import inspect

    from tools.distraction_server import mcp as distraction_mcp
    from tools.iterative_refinement_server import mcp as iterative_mcp
    from tools.repeat_character_server import mcp as repeat_mcp

    def create_mcp_tool_class(t_name: str, t_desc: str, t_inputs: Dict, t_func, t_params: List[str]):
        """Create a Tool class with proper forward signature."""
        # Build forward function with explicit parameters
        params_str = ", ".join(t_params) if t_params else ""
        kwargs_str = ", ".join([f"{p}={p}" for p in t_params]) if t_params else ""

        code = f"""
def forward(self{', ' + params_str if params_str else ''}):
    import asyncio
    result = _func({kwargs_str})
    if asyncio.iscoroutine(result):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(result)
        finally:
            loop.close()
    return result
"""
        local_vars = {"_func": t_func}
        exec(code, local_vars)

        class MCPTool(Tool):
            name = t_name
            description = t_desc
            inputs = t_inputs
            output_type = "string"

        MCPTool.forward = local_vars["forward"]
        return MCPTool

    tools = []

    # Collect all MCP server instances
    mcp_servers = [distraction_mcp, iterative_mcp, repeat_mcp]

    for mcp_server in mcp_servers:
        # Access registered tools from FastMCP
        for tool_name, tool_func in mcp_server._tool_manager._tools.items():
            # Extract function metadata
            fn = tool_func.fn
            description = fn.__doc__ or f"Tool: {tool_name}"
            description = description[:500] if len(description) > 500 else description

            # Parse function signature for inputs
            sig = inspect.signature(fn)
            inputs = {}
            param_names = []
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                param_type = "string"
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == float:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif hasattr(param.annotation, "__origin__"):
                        # Handle List, Dict, etc.
                        param_type = "any"
                inputs[param_name] = {
                    "type": param_type,
                    "description": f"Parameter: {param_name}",
                }
                param_names.append(param_name)

            tool_class = create_mcp_tool_class(tool_name, description, inputs, fn, param_names)
            tools.append(tool_class())

    return tools


def sanitize_tool_name(name: str) -> str:
    """Sanitize a tool name to be a valid Python identifier."""
    clean_name = "".join(c if c.isalnum() else "_" for c in name)
    if clean_name and clean_name[0].isdigit():
        clean_name = "tool_" + clean_name
    return clean_name or f"tool_{hash(name) % 100000}"


def create_tool_class(
    name: str,
    description: str,
    original_id: str | None = None,
    *,
    executor: ToolRetExecutor | None = None,
) -> type[Tool]:
    """Dynamically create a Tool subclass."""

    class DynamicTool(Tool):
        pass

    DynamicTool.name = name
    DynamicTool.description = description
    DynamicTool.inputs = {"query": {"type": "string", "description": "Input for this tool"}}
    DynamicTool.output_type = "string"

    def forward(self, query: str) -> str:
        header = f"[{name}] Processed: {query}"
        result = executor(original_id, query)
        return f"{header}\n{result}"

    DynamicTool.forward = forward

    if original_id:
        DynamicTool._original_id = original_id

    return DynamicTool


def load_tools_from_jsonl(
    jsonl_path: str,
    max_description_length: int = 150,
    *,
    executor: ToolRetExecutor | None = None,
) -> List[Tool]:
    """
    Load tools from JSONL and convert to smolagents Tool instances.

    Args:
        jsonl_path: Path to JSONL file containing tool definitions
        max_description_length: Maximum length for tool descriptions
        executor: Optional callable receiving the original ToolRet ID and query

    Returns:
        List of Tool instances
    """
    tools = []
    tool_name_counter: Dict[str, int] = defaultdict(int)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            tool_data = json.loads(line)
            doc = json.loads(tool_data["documentation"])
            original_id = tool_data["id"]

            base_name = sanitize_tool_name(doc.get("name", original_id))

            # Make name unique
            tool_name_counter[base_name] += 1
            if tool_name_counter[base_name] > 1:
                tool_name = f"{base_name}_{tool_name_counter[base_name]}"
            else:
                tool_name = base_name

            # Truncate description
            description = doc.get("description", "No description available")
            if len(description) > max_description_length:
                description = description[:max_description_length] + "..."

            tool_class = create_tool_class(tool_name, description, original_id, executor=executor)
            tools.append(tool_class())

    return tools


def get_tool_category(tool: Tool) -> str:
    """Extract category from tool ID."""
    tool_id = getattr(tool, "_original_id", tool.name)
    parts = tool_id.split("_")
    return "_".join(parts[:-1]) if parts[-1].isdigit() else tool_id


def sample_tools_by_category(
    tools: List[Tool],
    n_per_category: int,
    max_total: int,
    seed: int = 42,
) -> List[Tool]:
    """
    Sample tools by category with limits.

    Args:
        tools: List of all tools
        n_per_category: Number of tools to sample per category
        max_total: Maximum total tools
        seed: Random seed for reproducibility

    Returns:
        Sampled list of tools
    """
    if seed is not None:
        random.seed(seed)

    # Group by category
    grouped: Dict[str, List[Tool]] = defaultdict(list)
    for tool in tools:
        grouped[get_tool_category(tool)].append(tool)

    # Sample from each category
    sampled = []
    for category_tools in grouped.values():
        sample_size = min(n_per_category, len(category_tools))
        sampled.extend(random.sample(category_tools, sample_size))

    # Limit total
    if len(sampled) > max_total:
        sampled = random.sample(sampled, max_total)

    return sampled
