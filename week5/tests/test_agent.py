"""Unit tests for the Agent class.

All tests use MockLLMClient (zero API cost) and an EchoTool that doesn't
touch the real techcorp.db. Real-data integration tests live elsewhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from app.agent import Agent, FunctionCall, LLMResponse
from tests.conftest import make_text_response, make_tool_call_response


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------
def test_agent_init_with_injected_client(mock_llm, echo_tool, memory_storage):
    a = Agent(
        tools=[echo_tool], storage=memory_storage, llm_client=mock_llm,
        max_iterations=3,
    )
    assert "echo" in a.tools_by_name
    assert a.max_iterations == 3
    assert a.llm is mock_llm


def test_agent_init_without_client_or_key_raises(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises((ValueError, RuntimeError)):
        # No llm_client, no api_key -> tries to build GeminiClient -> empty key
        Agent(tools=[], llm_client=None, api_key="")


# ----------------------------------------------------------------------
# Single-shot: LLM answers directly without calling tools
# ----------------------------------------------------------------------
def test_single_shot_no_tool_calls(mock_llm, echo_tool, memory_storage):
    mock_llm.responses = [make_text_response("Direct answer", in_tok=80, out_tok=20)]
    a = Agent(tools=[echo_tool], storage=memory_storage, llm_client=mock_llm)
    result = a.query("hello")
    assert result["answer"] == "Direct answer"
    assert result["iterations"] == 1
    assert result["input_tokens"] == 80
    assert result["output_tokens"] == 20
    assert result["tokens_used"] == 100
    assert result["cost"] > 0
    assert result["tool_calls"] == []
    assert len(mock_llm.call_log) == 1


# ----------------------------------------------------------------------
# Multi-shot: tool call then final answer
# ----------------------------------------------------------------------
def test_two_shot_tool_then_answer(mock_llm, echo_tool, memory_storage):
    mock_llm.responses = [
        make_tool_call_response(("echo", {"value": "hi"}), in_tok=100, out_tok=15),
        make_text_response("The tool said: hi", in_tok=120, out_tok=20),
    ]
    a = Agent(tools=[echo_tool], storage=memory_storage, llm_client=mock_llm)
    result = a.query("call echo with hi")
    assert result["iterations"] == 2
    assert result["input_tokens"] == 220
    assert result["output_tokens"] == 35
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool"] == "echo"
    assert result["tool_calls"][0]["args"] == {"value": "hi"}
    assert result["answer"] == "The tool said: hi"


# ----------------------------------------------------------------------
# Multiple tools called in one LLM turn (parallel)
# ----------------------------------------------------------------------
def test_parallel_tool_calls_in_single_turn(mock_llm, echo_tool, memory_storage):
    mock_llm.responses = [
        make_tool_call_response(
            ("echo", {"value": "a"}),
            ("echo", {"value": "b"}),
            in_tok=100, out_tok=30,
        ),
        make_text_response("ok", in_tok=150, out_tok=10),
    ]
    a = Agent(tools=[echo_tool], storage=memory_storage, llm_client=mock_llm)
    result = a.query("call echo twice")
    assert result["iterations"] == 2
    assert len(result["tool_calls"]) == 2


# ----------------------------------------------------------------------
# Chained tools across multiple iterations
# ----------------------------------------------------------------------
def test_chained_tool_calls(mock_llm, echo_tool, memory_storage):
    mock_llm.responses = [
        make_tool_call_response(("echo", {"value": "first"})),
        make_tool_call_response(("echo", {"value": "second"})),
        make_text_response("final answer"),
    ]
    a = Agent(
        tools=[echo_tool], storage=memory_storage, llm_client=mock_llm,
        max_iterations=5,
    )
    result = a.query("chain two tools")
    assert result["iterations"] == 3
    assert len(result["tool_calls"]) == 2
    assert result["answer"] == "final answer"


# ----------------------------------------------------------------------
# Unknown tool -> agent reports error in tool result, LLM can recover
# ----------------------------------------------------------------------
def test_unknown_tool_returns_error_to_llm(mock_llm, echo_tool, memory_storage):
    mock_llm.responses = [
        make_tool_call_response(("nonexistent_tool", {"x": 1})),
        make_text_response("I cannot do that"),
    ]
    a = Agent(tools=[echo_tool], storage=memory_storage, llm_client=mock_llm)
    result = a.query("test")
    # The unknown call IS recorded in tool_calls (with error result)
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool"] == "nonexistent_tool"
    assert result["answer"] == "I cannot do that"


# ----------------------------------------------------------------------
# Bad args -> agent reports error in result; LLM can recover
# ----------------------------------------------------------------------
def test_bad_tool_args_does_not_crash(mock_llm, echo_tool, memory_storage):
    mock_llm.responses = [
        # echo's signature is (value: str); pass an unknown kwarg
        make_tool_call_response(("echo", {"wrong_arg": "x"})),
        make_text_response("recovered"),
    ]
    a = Agent(tools=[echo_tool], storage=memory_storage, llm_client=mock_llm)
    result = a.query("bad args")
    assert result["iterations"] == 2
    assert result["answer"] == "recovered"


# ----------------------------------------------------------------------
# Iteration cap
# ----------------------------------------------------------------------
def test_max_iterations_reached(mock_llm, echo_tool, memory_storage):
    # LLM keeps calling tools forever
    mock_llm.responses = [
        make_tool_call_response(("echo", {"value": "x"})) for _ in range(10)
    ]
    a = Agent(
        tools=[echo_tool], storage=memory_storage, llm_client=mock_llm,
        max_iterations=3,
    )
    result = a.query("infinite loop")
    assert result["iterations"] == 3
    assert result.get("warning") == "max_iterations_reached"
    assert len(result["tool_calls"]) == 3


# ----------------------------------------------------------------------
# LLM exception is captured as error result, not propagated
# ----------------------------------------------------------------------
def test_llm_exception_handled(echo_tool, memory_storage):
    class BrokenClient:
        def generate(self, messages, tools, model):
            raise RuntimeError("API down")
    a = Agent(
        tools=[echo_tool], storage=memory_storage, llm_client=BrokenClient(),
    )
    result = a.query("doesn't matter")
    assert "error" in result
    assert "API down" in result["error"]
    assert "(agent error" in result["answer"]


# ----------------------------------------------------------------------
# Storage / metrics
# ----------------------------------------------------------------------
def test_query_recorded_in_storage(mock_llm, echo_tool, memory_storage):
    mock_llm.responses = [make_text_response("ok", in_tok=50, out_tok=25)]
    a = Agent(tools=[echo_tool], storage=memory_storage, llm_client=mock_llm)
    a.query("test", user_id="alice", user_role="engineer")
    history = memory_storage.query("query_history")
    assert len(history) == 1
    assert history[0]["user_id"] == "alice"
    assert history[0]["user_role"] == "engineer"
    assert history[0]["query"] == "test"
    assert history[0]["cost"] > 0


def test_metrics_roll_up_across_queries(mock_llm, echo_tool, memory_storage):
    mock_llm.responses = [
        make_text_response("a", in_tok=100, out_tok=50),
        make_text_response("b", in_tok=200, out_tok=80),
        make_text_response("c", in_tok=150, out_tok=60),
    ]
    a = Agent(tools=[echo_tool], storage=memory_storage, llm_client=mock_llm)
    a.query("q1")
    a.query("q2")
    a.query("q3")
    m = a.get_metrics()
    assert m["total_queries"] == 3
    assert m["total_input_tokens"] == 450
    assert m["total_output_tokens"] == 190
    assert m["total_cost"] > 0
    assert m["avg_cost_per_query"] == m["total_cost"] / 3


def test_metrics_empty_history():
    """get_metrics on a fresh agent returns zeros, not divide-by-zero error."""
    from app.storage import InMemoryStorage
    a = Agent(
        tools=[], storage=InMemoryStorage(),
        llm_client=type("X", (), {"generate": lambda *a, **k: None})(),
    )
    m = a.get_metrics()
    assert m["total_queries"] == 0
    assert m["total_cost"] == 0.0
    assert m["avg_cost_per_query"] == 0.0


# ----------------------------------------------------------------------
# Cost calculation correctness
# ----------------------------------------------------------------------
def test_cost_calculation_pro_pricing(mock_llm, echo_tool, memory_storage):
    # Pro: $0.075 input / 1M, $0.30 output / 1M
    # 1M input + 1M output should cost $0.375
    mock_llm.responses = [make_text_response(
        "answer", in_tok=1_000_000, out_tok=1_000_000,
    )]
    a = Agent(
        tools=[echo_tool], storage=memory_storage, llm_client=mock_llm,
        model="gemini-2.5-pro",
    )
    result = a.query("test")
    assert result["cost"] == pytest.approx(0.375, rel=1e-6)


def test_cost_calculation_flash_cheaper(mock_llm, echo_tool, memory_storage):
    """Flash should cost less than Pro for the same token counts."""
    pro_llm = type(mock_llm)(responses=[make_text_response("a", 1000, 1000)])
    flash_llm = type(mock_llm)(responses=[make_text_response("a", 1000, 1000)])
    pro_a = Agent(tools=[], storage=memory_storage, llm_client=pro_llm, model="gemini-2.5-pro")
    flash_a = Agent(tools=[], storage=memory_storage, llm_client=flash_llm, model="gemini-1.5-flash")
    assert flash_a.query("t")["cost"] < pro_a.query("t")["cost"]
