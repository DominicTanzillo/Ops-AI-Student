"""Shared test fixtures + the MockLLMClient used by agent tests."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from app.agent import FunctionCall, LLMResponse  # noqa: E402
from app.storage import InMemoryStorage  # noqa: E402
from app.tools import Tool  # noqa: E402


# ----------------------------------------------------------------------
# MockLLMClient: returns canned responses in sequence
# ----------------------------------------------------------------------
@dataclass
class MockLLMClient:
    """Test double for the LLMClient Protocol.

    Configure with a sequence of LLMResponse objects (or callables that
    take messages+tools and return one). Each call to .generate() pops the
    next response. Records all calls for inspection.
    """

    responses: list = field(default_factory=list)
    call_log: list[dict[str, Any]] = field(default_factory=list)

    def generate(self, messages, tools, model) -> LLMResponse:
        self.call_log.append({
            "messages": list(messages),
            "tool_names": [t.name for t in tools],
            "model": model,
        })
        if not self.responses:
            raise AssertionError(
                f"MockLLMClient exhausted - test made {len(self.call_log)} calls "
                f"but only {len(self.responses)} responses were configured."
            )
        nxt = self.responses.pop(0)
        if callable(nxt):
            return nxt(messages, tools)
        return nxt


def make_text_response(text: str, in_tok: int = 100, out_tok: int = 50) -> LLMResponse:
    return LLMResponse(text=text, input_tokens=in_tok, output_tokens=out_tok)


def make_tool_call_response(
    *calls: tuple[str, dict[str, Any]],
    in_tok: int = 100,
    out_tok: int = 20,
) -> LLMResponse:
    return LLMResponse(
        function_calls=[FunctionCall(name=n, args=a) for n, a in calls],
        input_tokens=in_tok,
        output_tokens=out_tok,
    )


# ----------------------------------------------------------------------
# A trivial echo tool for tests that don't want to hit the real db/json
# ----------------------------------------------------------------------
class _EchoTool(Tool):
    name = "echo"
    description = "Echo back arguments."
    parameters_schema = {
        "type": "object",
        "properties": {
            "value": {"type": "string", "description": "string to echo"},
        },
        "required": ["value"],
    }

    def execute(self, value: str = "") -> str:
        return f'{{"echoed": "{value}"}}'


@pytest.fixture
def echo_tool() -> Tool:
    return _EchoTool()


@pytest.fixture
def mock_llm() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture
def memory_storage() -> InMemoryStorage:
    return InMemoryStorage()
