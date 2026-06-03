"""Integration tests: Week 5 agent + Week 6 guardrails.

Verifies the four integration points:
  1. Rate limit pre-check rejects with structured response, no LLM call
  2. Budget pre-check rejects with structured response, no LLM call
  3. Cost is added to user_spending after a successful query
  4. Response is redacted via AccessController before return

All using MockLLMClient -> zero cost.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from app.access_control import AccessController, CostEnforcer, RateLimiter
from app.agent import Agent
from app.storage import InMemoryStorage
from tests.conftest import MockLLMClient, make_text_response


@pytest.fixture
def policy_file(tmp_path):
    policy = {
        "roles": {
            "engineer": {"permissions": {"view_employee_directory": True}},
            "hr":       {"permissions": {"view_employee_directory": True}},
        },
        "sensitive_fields": {
            "salary": {"visibility": ["hr"], "redact": True},
        },
    }
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(policy))
    return p


@pytest.fixture
def storage():
    return InMemoryStorage()


@pytest.fixture
def guarded_agent(policy_file, storage):
    """Agent with all three guardrails wired in, plus a mock LLM."""
    mock = MockLLMClient(responses=[make_text_response("OK", in_tok=100, out_tok=30)])
    ac = AccessController(policy_file, storage=storage)
    rl = RateLimiter(max_queries_per_minute=2, storage=storage)
    ce = CostEnforcer(role_budgets={"engineer": 100.0, "hr": 200.0}, storage=storage)
    agent = Agent(
        tools=[], storage=storage, llm_client=mock,
        access_controller=ac, rate_limiter=rl, cost_enforcer=ce,
    )
    return agent, mock, ac, rl, ce, storage


# ----------------------------------------------------------------------
# 1. Rate-limit pre-check
# ----------------------------------------------------------------------
def test_rate_limit_blocks_third_query(guarded_agent):
    agent, mock, _, rl, _, _ = guarded_agent
    # Cap is 2; need 2 mock responses for the 2 allowed queries
    mock.responses.append(make_text_response("OK2", in_tok=100, out_tok=30))
    r1 = agent.query("q1", user_id="alice", user_role="engineer")
    r2 = agent.query("q2", user_id="alice", user_role="engineer")
    r3 = agent.query("q3", user_id="alice", user_role="engineer")
    assert r1.get("rejected") is None
    assert r2.get("rejected") is None
    assert r3["rejected"] == "rate_limit_exceeded"
    # The rate-limited query did NOT call the LLM
    assert len(mock.call_log) == 2


def test_rate_limit_per_user(guarded_agent):
    """Alice hits cap; Bob still allowed."""
    agent, mock, _, _, _, _ = guarded_agent
    mock.responses.extend([
        make_text_response("alice2"),
        make_text_response("bob1"),
    ])
    agent.query("q", user_id="alice", user_role="engineer")
    agent.query("q", user_id="alice", user_role="engineer")
    alice_blocked = agent.query("q", user_id="alice", user_role="engineer")
    bob_ok = agent.query("q", user_id="bob", user_role="engineer")
    assert alice_blocked["rejected"] == "rate_limit_exceeded"
    assert bob_ok.get("rejected") is None


# ----------------------------------------------------------------------
# 2. Budget pre-check
# ----------------------------------------------------------------------
def test_budget_blocks_when_over_cap(policy_file, storage):
    mock = MockLLMClient(responses=[])  # should not be called
    ac = AccessController(policy_file, storage=storage)
    rl = RateLimiter(max_queries_per_minute=999, storage=storage)
    ce = CostEnforcer(role_budgets={"engineer": 0.005}, storage=storage)  # tiny budget
    agent = Agent(
        tools=[], storage=storage, llm_client=mock,
        access_controller=ac, rate_limiter=rl, cost_enforcer=ce,
    )
    result = agent.query("q", user_id="alice", user_role="engineer")
    assert result["rejected"] == "budget_exceeded"
    # LLM was never called
    assert len(mock.call_log) == 0


def test_budget_check_uses_role_specific_budget(policy_file, storage):
    """Same user, two roles: one has budget, the other doesn't."""
    mock = MockLLMClient(responses=[make_text_response("ok"), make_text_response("ok2")])
    ac = AccessController(policy_file, storage=storage)
    rl = RateLimiter(max_queries_per_minute=999, storage=storage)
    ce = CostEnforcer(role_budgets={"engineer": 0.001, "hr": 1000.0}, storage=storage)
    agent = Agent(
        tools=[], storage=storage, llm_client=mock,
        access_controller=ac, rate_limiter=rl, cost_enforcer=ce,
    )
    # As engineer: budget = $0.001 < preflight $0.01 -> reject
    r1 = agent.query("q", user_id="alice", user_role="engineer")
    assert r1["rejected"] == "budget_exceeded"
    # Same user as hr: budget $1000 -> allowed
    r2 = agent.query("q", user_id="alice", user_role="hr")
    assert r2.get("rejected") is None


# ----------------------------------------------------------------------
# 3. Cost tracking
# ----------------------------------------------------------------------
def test_cost_added_to_spending_after_success(guarded_agent):
    agent, _, _, _, ce, _ = guarded_agent
    result = agent.query("q1", user_id="alice", user_role="engineer")
    spent = ce.get_user_spending("alice")
    assert spent == pytest.approx(result["cost"])
    assert spent > 0  # the mock response had nonzero tokens


def test_cost_accumulates_across_queries(policy_file, storage):
    mock = MockLLMClient(responses=[
        make_text_response("a", in_tok=100, out_tok=50),
        make_text_response("b", in_tok=200, out_tok=80),
    ])
    ac = AccessController(policy_file, storage=storage)
    rl = RateLimiter(max_queries_per_minute=99, storage=storage)
    ce = CostEnforcer(role_budgets={"engineer": 100.0}, storage=storage)
    agent = Agent(
        tools=[], storage=storage, llm_client=mock,
        access_controller=ac, rate_limiter=rl, cost_enforcer=ce,
    )
    r1 = agent.query("a", user_id="alice", user_role="engineer")
    r2 = agent.query("b", user_id="alice", user_role="engineer")
    assert ce.get_user_spending("alice") == pytest.approx(r1["cost"] + r2["cost"])


def test_rejected_queries_do_not_cost(guarded_agent):
    agent, mock, _, rl, ce, _ = guarded_agent
    # Burn the rate limit
    mock.responses.append(make_text_response("ok"))
    agent.query("q", user_id="alice", user_role="engineer")
    agent.query("q", user_id="alice", user_role="engineer")
    # 3rd query is rate-limited; should add 0 cost
    cost_before = ce.get_user_spending("alice")
    agent.query("q", user_id="alice", user_role="engineer")
    cost_after = ce.get_user_spending("alice")
    assert cost_after == cost_before  # no charge for rejected query


# ----------------------------------------------------------------------
# 4. Response redaction
# ----------------------------------------------------------------------
def test_response_redacted_for_engineer(policy_file, storage):
    mock = MockLLMClient(responses=[make_text_response(
        '{"name": "Alice", "salary": 200000}',
        in_tok=100, out_tok=30,
    )])
    ac = AccessController(policy_file, storage=storage)
    agent = Agent(
        tools=[], storage=storage, llm_client=mock,
        access_controller=ac,
    )
    result = agent.query("show me alice's salary", user_role="engineer")
    assert "[REDACTED]" in result["answer"]
    assert "200000" not in result["answer"]


def test_response_not_redacted_for_hr(policy_file, storage):
    mock = MockLLMClient(responses=[make_text_response(
        '{"name": "Alice", "salary": 200000}',
        in_tok=100, out_tok=30,
    )])
    ac = AccessController(policy_file, storage=storage)
    agent = Agent(
        tools=[], storage=storage, llm_client=mock,
        access_controller=ac,
    )
    result = agent.query("show me alice's salary", user_role="hr")
    assert "200000" in result["answer"]
    assert "[REDACTED]" not in result["answer"]


# ----------------------------------------------------------------------
# 5. Without guardrails, behavior matches Week 5
# ----------------------------------------------------------------------
def test_unguarded_agent_behaves_like_week5(storage):
    mock = MockLLMClient(responses=[make_text_response(
        '{"salary": 999}', in_tok=100, out_tok=10,
    )])
    agent = Agent(tools=[], storage=storage, llm_client=mock)  # no guardrails
    result = agent.query("q", user_role="engineer")
    assert "999" in result["answer"]  # not redacted (no AccessController)
    assert "rejected" not in result  # not blocked (no RateLimiter / CostEnforcer)
