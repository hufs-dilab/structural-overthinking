"""Run one repository MCP tool server over streamable HTTP for agent CLIs."""

from __future__ import annotations

import argparse
import importlib


TOOL_SERVER_MODULES = {
    "distraction": "tools.distraction_server",
    "iterative_refinement": "tools.iterative_refinement_server",
    "repeat_character": "tools.repeat_character_server",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local MCP tool server over HTTP.")
    parser.add_argument("--tool-server", required=True, choices=sorted(TOOL_SERVER_MODULES))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--path", default="/mcp")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    module = importlib.import_module(TOOL_SERVER_MODULES[args.tool_server])
    module.mcp.run(
        transport="http",
        host=args.host,
        port=args.port,
        path=args.path,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
