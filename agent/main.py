"""CLI entry point for the MLOps AI Agent.

Supports two modes:
    * **interactive** – a REPL loop that reads queries until the user types
      ``exit`` or ``quit``.
    * **single** – executes a single query supplied via ``--query`` and exits.

Usage::

    python -m agent.main --mode interactive
    python -m agent.main --mode single --query "Show me all MLflow experiments"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from .orchestrator import MLOpsAgent
from .schemas import AgentRequest

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------


def _pretty_print(response) -> None:  # noqa: ANN001
    """Print an :class:`AgentResponse` in a human-friendly format."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Request ID : {response.request_id}")
    print(f"  Confidence : {response.confidence}")
    print(f"{sep}")
    print(f"\n{response.response_text}\n")

    if response.warnings:
        print("[!] Warnings:")
        for warning in response.warnings:
            print(f"   * {warning}")
        print()

    if response.tool_results:
        dash_line = "-" * 40
        print(dash_line)
        print(f"  Tool Results ({len(response.tool_results)} total):")
        for tr in response.tool_results:
            status = "[OK]" if tr.success else "[FAIL]"
            print(f"    {status} {tr.tool_name} ({tr.duration_seconds}s)")
            if tr.error:
                print(f"      Error: {tr.error}")
        print(f"{dash_line}\n")


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------


async def _interactive_mode(agent: MLOpsAgent) -> None:
    """Run an interactive REPL loop."""
    print("\n=== MLOps AI Agent - Interactive Mode ===")
    print("   Type your query and press Enter. Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        request = AgentRequest(user_input=user_input)
        logger.info("Request created: %s", request.request_id)

        response = await agent.process_request(request)
        _pretty_print(response)


async def _single_mode(agent: MLOpsAgent, query: str) -> None:
    """Execute a single query and exit."""
    request = AgentRequest(user_input=query)
    logger.info("Single-query request: %s", request.request_id)

    response = await agent.process_request(request)
    _pretty_print(response)


# ---------------------------------------------------------------------------
# Argument parsing & main
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="mlops-agent",
        description="MLOps AI Agent – query DVC, MLflow, Airflow, and Kubernetes from a single CLI.",
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "single"],
        default="interactive",
        help="Execution mode (default: interactive).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query string (required when --mode=single).",
    )
    return parser


def main() -> None:
    """Entry point invoked by ``python -m agent.main``."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.mode == "single" and not args.query:
        parser.error("--query is required when using --mode=single")

    agent = MLOpsAgent()

    if args.mode == "interactive":
        asyncio.run(_interactive_mode(agent))
    else:
        asyncio.run(_single_mode(agent, args.query))


if __name__ == "__main__":
    main()
