"""Sanitized test fixtures for the host delegation capability boundary.

The installed host currently accepts a batch-only ``delegate_task`` schema. Tests
that need a nested conductor must declare depth >= 2; tests that validate a model
parameter use the separately declared schema rather than pretending the installed
host supports it.
"""

from contextlib import contextmanager
from copy import deepcopy
from typing import Iterator
from unittest.mock import patch

DIRECT_CLAUDE_TOOLS = ("delegate_task", "delegate_claude")
DEFERRED_CLAUDE_TOOLS = ("delegate_task", "tool_search", "tool_describe", "tool_call")

_BATCH_TASK_PROPERTIES = {
    "goal": {"type": "string"},
    "context": {"type": "string"},
    "role": {"type": "string"},
}


def delegate_task_schema(capability: str = "batch-only") -> dict:
    """Return a sanitized delegate_task schema for one declared host capability."""
    if capability not in {"batch-only", "model-param"}:
        raise ValueError(f"unknown delegate_task capability: {capability}")
    task_properties = deepcopy(_BATCH_TASK_PROPERTIES)
    properties = {
        "tasks": {
            "type": "array",
            "minItems": 1,
            "maxItems": 1,
            "items": {
                "type": "object",
                "properties": task_properties,
                "required": ["goal", "context"],
            },
        }
    }
    if capability == "model-param":
        # This is an explicit capability fixture, not a claim about the installed
        # batch-only host. The router detects a model parameter at this schema level.
        properties["model"] = {"type": "string", "enum": ["terra", "sonnet5"]}
    return {"type": "object", "properties": properties, "required": ["tasks"]}


def delegate_task_request(capability: str = "batch-only", *, wire: str = "openai") -> dict:
    """Build a request that exposes the selected schema in a real supported wire shape."""
    schema = delegate_task_schema(capability)
    if wire == "openai":
        tool = {"type": "function", "name": "delegate_task", "parameters": schema}
    elif wire == "anthropic":
        tool = {"name": "delegate_task", "input_schema": schema}
    else:
        raise ValueError(f"unknown tool wire: {wire}")
    return {"tools": [tool]}


@contextmanager
def host_delegation(*, depth: int, orchestrator_enabled: bool = True, max_concurrent_children: int = 3,
                    max_iterations: int = 250) -> Iterator[None]:
    """Patch the host settings that determine direct-worker and conductor topology."""
    if depth < 1:
        raise ValueError("depth must allow direct workers")
    with patch("tools.delegate_tool_config._get_max_spawn_depth", return_value=depth), \
         patch("tools.delegate_tool_config._get_orchestrator_enabled", return_value=orchestrator_enabled), \
         patch("tools.delegate_tool_config._get_max_concurrent_children", return_value=max_concurrent_children), \
         patch("tools.delegate_tool_config._load_config", return_value={"max_iterations": max_iterations}):
        yield
