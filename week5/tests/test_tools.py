"""Unit tests for the 3 agent tools.

These run against the real techcorp.db + json files. Zero LLM, zero cost.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from app.tools import (
    EmployeeLookupTool,
    ExpenseQueryTool,
    PolicySearchTool,
    Tool,
    default_tools,
)

REAL_DB = HERE.parent / "data" / "techcorp.db"
REAL_DOCS = HERE.parent / "data" / "documents.json"
REAL_POLICIES = HERE.parent / "data" / "policies.json"


# ----------------------------------------------------------------------
# Tool base class
# ----------------------------------------------------------------------
def test_base_tool_execute_raises():
    t = Tool()
    with pytest.raises(NotImplementedError):
        t.execute()


def test_function_declaration_shape():
    tool = PolicySearchTool(documents_path=REAL_DOCS)
    fd = tool.to_function_declaration()
    assert fd["name"] == "policy_search"
    assert "description" in fd
    assert fd["parameters"]["type"] == "object"
    assert "query" in fd["parameters"]["properties"]
    assert fd["parameters"]["required"] == ["query"]


# ----------------------------------------------------------------------
# EmployeeLookupTool
# ----------------------------------------------------------------------
@pytest.mark.skipif(not REAL_DB.exists(), reason="techcorp.db not present")
def test_employee_lookup_by_name_returns_matches():
    t = EmployeeLookupTool(db_path=REAL_DB)
    # 'Smith' is a common surname; should match many employees
    out = json.loads(t.execute(employee_name="Smith"))
    assert "matches" in out
    assert isinstance(out["matches"], list)
    assert len(out["matches"]) > 0
    assert len(out["matches"]) <= 5  # default limit
    assert "name" in out["matches"][0]
    assert "salary" in out["matches"][0]  # not redacted at tool layer


@pytest.mark.skipif(not REAL_DB.exists(), reason="techcorp.db not present")
def test_employee_lookup_by_id():
    t = EmployeeLookupTool(db_path=REAL_DB)
    out = json.loads(t.execute(employee_id=1))
    assert out["matches"][0]["id"] == 1


@pytest.mark.skipif(not REAL_DB.exists(), reason="techcorp.db not present")
def test_employee_lookup_no_match_returns_empty():
    t = EmployeeLookupTool(db_path=REAL_DB)
    out = json.loads(t.execute(employee_name="ZZZ_NOBODY_HAS_THIS_NAME_ZZZ"))
    assert out["matches"] == []
    assert "no employees" in out["note"].lower()


def test_employee_lookup_missing_args_returns_error():
    t = EmployeeLookupTool(db_path=str(REAL_DB))
    out = json.loads(t.execute())
    assert "error" in out


@pytest.mark.skipif(not REAL_DB.exists(), reason="techcorp.db not present")
def test_employee_lookup_respects_limit():
    t = EmployeeLookupTool(db_path=REAL_DB)
    out = json.loads(t.execute(employee_name="a", limit=2))
    assert len(out["matches"]) <= 2


# ----------------------------------------------------------------------
# PolicySearchTool
# ----------------------------------------------------------------------
@pytest.mark.skipif(not REAL_DOCS.exists(), reason="documents.json not present")
def test_policy_search_finds_relevant_docs():
    t = PolicySearchTool(documents_path=REAL_DOCS)
    out = json.loads(t.execute(query="employee handbook"))
    assert out["results"], "should find at least the handbook"
    titles = [r["title"].lower() for r in out["results"]]
    assert any("handbook" in t for t in titles), \
        f"expected a handbook doc in results; got titles={titles}"


@pytest.mark.skipif(not REAL_DOCS.exists(), reason="documents.json not present")
def test_policy_search_empty_query_errors():
    t = PolicySearchTool(documents_path=REAL_DOCS)
    out = json.loads(t.execute(query=""))
    assert "error" in out


@pytest.mark.skipif(not REAL_DOCS.exists(), reason="documents.json not present")
def test_policy_search_no_match_returns_empty():
    t = PolicySearchTool(documents_path=REAL_DOCS)
    out = json.loads(t.execute(query="xqzj_definitely_not_in_any_policy"))
    assert out["results"] == []


@pytest.mark.skipif(not REAL_DOCS.exists(), reason="documents.json not present")
def test_policy_search_category_filter():
    t = PolicySearchTool(documents_path=REAL_DOCS)
    # 'policy' should match many docs; filter to HR only
    out = json.loads(t.execute(query="policy", category="HR"))
    for r in out["results"]:
        # category lookup is case-insensitive; original may be 'HR'/'hr'/etc
        assert r["category"].lower() == "hr", f"got category={r['category']}"


@pytest.mark.skipif(not REAL_DOCS.exists(), reason="documents.json not present")
def test_policy_search_respects_limit():
    t = PolicySearchTool(documents_path=REAL_DOCS)
    out = json.loads(t.execute(query="employee", limit=2))
    assert len(out["results"]) <= 2


# ----------------------------------------------------------------------
# ExpenseQueryTool
# ----------------------------------------------------------------------
@pytest.mark.skipif(not REAL_POLICIES.exists(), reason="policies.json not present")
def test_expense_query_approval_limits():
    t = ExpenseQueryTool(policies_path=REAL_POLICIES)
    out = json.loads(t.execute(topic="approval_limits"))
    assert "approval_limits" in out
    # Sanity: manager limit > IC limit
    al = out["approval_limits"]
    assert al["manager"] > al["ic1_ic2"]


@pytest.mark.skipif(not REAL_POLICIES.exists(), reason="policies.json not present")
def test_expense_query_approval_limit_for_role():
    t = ExpenseQueryTool(policies_path=REAL_POLICIES)
    out = json.loads(t.execute(topic="approval_limits", role="manager"))
    assert out["role"] == "manager"
    assert isinstance(out["approval_limit"], (int, float))


@pytest.mark.skipif(not REAL_POLICIES.exists(), reason="policies.json not present")
def test_expense_query_travel():
    t = ExpenseQueryTool(policies_path=REAL_POLICIES)
    out = json.loads(t.execute(topic="travel_limits"))
    assert "travel" in out
    assert "budget_limits" in out["travel"]


@pytest.mark.skipif(not REAL_POLICIES.exists(), reason="policies.json not present")
def test_expense_query_per_diem():
    t = ExpenseQueryTool(policies_path=REAL_POLICIES)
    out = json.loads(t.execute(topic="per_diem"))
    for k in ("meal_breakfast", "meal_lunch", "meal_dinner"):
        assert k in out
        assert isinstance(out[k], (int, float))


@pytest.mark.skipif(not REAL_POLICIES.exists(), reason="policies.json not present")
def test_expense_query_pto():
    t = ExpenseQueryTool(policies_path=REAL_POLICIES)
    out = json.loads(t.execute(topic="pto"))
    assert "pto" in out


@pytest.mark.skipif(not REAL_POLICIES.exists(), reason="policies.json not present")
def test_expense_query_unknown_topic_lists_valid():
    t = ExpenseQueryTool(policies_path=REAL_POLICIES)
    out = json.loads(t.execute(topic="totally_made_up"))
    assert "error" in out
    assert "available_topics" in out


# ----------------------------------------------------------------------
# default_tools factory
# ----------------------------------------------------------------------
@pytest.mark.skipif(not REAL_DB.exists(), reason="techcorp.db not present")
def test_default_tools_returns_three_tools():
    tools = default_tools()
    assert len(tools) == 3
    names = {t.name for t in tools}
    assert names == {"employee_lookup", "policy_search", "expense_query"}
    # Each tool advertises a function declaration
    for t in tools:
        fd = t.to_function_declaration()
        assert fd["name"]
        assert fd["description"]
        assert "parameters" in fd
