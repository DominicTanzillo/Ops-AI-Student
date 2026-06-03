"""Unit tests for AccessController, RateLimiter, CostEnforcer.

Pure-Python guardrails - no LLM, no Gemini, no network. All zero-cost.
Uses Week 5's access_control.json (copied via the test fixture).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from app.access_control import (
    REDACTED,
    AccessController,
    CostEnforcer,
    RateLimiter,
)
from app.storage import InMemoryStorage

REAL_POLICY = HERE.parent / "data" / "access_control.json"


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def storage():
    return InMemoryStorage()


@pytest.fixture
def policy_file(tmp_path):
    """Self-contained access policy with the fields and roles we test."""
    policy = {
        "roles": {
            "engineer": {"permissions": {"view_employee_directory": True, "view_hr_data": False}},
            "manager":  {"permissions": {"view_employee_directory": True, "view_hr_data": False, "view_other_salaries": True}},
            "hr":       {"permissions": {"view_employee_directory": True, "view_hr_data": True}},
            "finance":  {"permissions": {"view_financial_reports": True}},
            "executive": {"permissions": {"view_financial_reports": True, "view_hr_data": True}},
        },
        "sensitive_fields": {
            "salary": {"visibility": ["executive", "hr", "finance"], "redact": True},
            "ssn":    {"visibility": ["hr", "finance"], "redact": True},
            "address": {"visibility": ["hr", "executive"], "redact": True},
        },
    }
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(policy))
    return p


@pytest.fixture
def controller(policy_file, storage):
    return AccessController(policy_file, storage=storage)


# ======================================================================
# AccessController
# ======================================================================
def test_init_loads_policy(controller):
    assert "engineer" in controller.policy["roles"]
    assert "salary" in controller.policy["sensitive_fields"]


def test_has_permission_true_when_granted(controller):
    assert controller.has_permission("engineer", "view_employee_directory") is True


def test_has_permission_false_when_denied(controller):
    assert controller.has_permission("engineer", "view_hr_data") is False


def test_has_permission_unknown_role(controller):
    assert controller.has_permission("intruder", "view_employee_directory") is False


def test_has_permission_unknown_capability(controller):
    assert controller.has_permission("engineer", "delete_universe") is False


# ----------------------------------------------------------------------
# can_view_field
# ----------------------------------------------------------------------
def test_can_view_salary_allowed_for_hr(controller):
    assert controller.can_view_field("hr", "salary") is True


def test_can_view_salary_denied_for_engineer(controller):
    assert controller.can_view_field("engineer", "salary") is False


def test_can_view_unknown_field_is_allowed(controller):
    # Non-sensitive fields default to allowed (otherwise the agent couldn't
    # show employee names).
    assert controller.can_view_field("engineer", "name") is True


def test_can_view_ssn_only_hr_or_finance(controller):
    for role, expected in [
        ("hr", True), ("finance", True), ("engineer", False),
        ("manager", False), ("executive", False),
    ]:
        assert controller.can_view_field(role, "ssn") == expected, role


# ----------------------------------------------------------------------
# Redaction
# ----------------------------------------------------------------------
def test_redact_response_replaces_salary_for_engineer(controller):
    response = '{"name": "Alice", "salary": 150000, "department": "Engineering"}'
    out = controller.redact_response("engineer", response)
    assert REDACTED in out
    assert "150000" not in out
    assert "Alice" in out  # non-sensitive fields preserved


def test_redact_response_keeps_salary_for_hr(controller):
    response = '{"name": "Alice", "salary": 150000}'
    out = controller.redact_response("hr", response)
    assert "150000" in out
    assert REDACTED not in out


def test_redact_handles_multiple_sensitive_fields(controller):
    response = '{"name": "Bob", "salary": 200000, "ssn": "123-45-6789", "address": "1 Main St"}'
    out = controller.redact_response("engineer", response)
    assert REDACTED in out
    assert "200000" not in out
    assert "123-45-6789" not in out
    assert "1 Main St" not in out


# ----------------------------------------------------------------------
# Audit log
# ----------------------------------------------------------------------
def test_audit_log_records_field_access(controller):
    controller.can_view_field("engineer", "salary")  # denied
    log = controller.get_audit_log()
    assert len(log) == 1
    assert log[0]["role"] == "engineer"
    assert log[0]["field"] == "salary"
    assert log[0]["allowed"] is False


def test_audit_log_records_document_access(controller):
    doc = {"id": "doc_001", "sensitivity": "Confidential"}
    controller.can_view_document("engineer", doc)  # denied
    log = controller.get_audit_log()
    assert any(e["resource"].endswith("doc_001") and not e["allowed"] for e in log)


def test_audit_log_persists_in_storage(controller, storage):
    controller.can_view_field("engineer", "salary")
    entries = storage.query("audit_log")
    assert len(entries) >= 1


# ----------------------------------------------------------------------
# Document visibility
# ----------------------------------------------------------------------
def test_can_view_document_public_allowed_for_engineer(controller):
    assert controller.can_view_document("engineer", {"id": "d", "sensitivity": "Public"}) is True


def test_can_view_document_confidential_denied_for_engineer(controller):
    assert controller.can_view_document("engineer", {"id": "d", "sensitivity": "Confidential"}) is False


def test_can_view_document_confidential_allowed_for_manager(controller):
    assert controller.can_view_document("manager", {"id": "d", "sensitivity": "Confidential"}) is True


def test_can_view_document_restricted_denied_for_manager(controller):
    assert controller.can_view_document("manager", {"id": "d", "sensitivity": "Restricted"}) is False


def test_filter_documents_returns_only_viewable(controller):
    docs = [
        {"id": "d1", "sensitivity": "Public"},
        {"id": "d2", "sensitivity": "Confidential"},
        {"id": "d3", "sensitivity": "Internal"},
        {"id": "d4", "sensitivity": "Restricted"},
    ]
    filtered = controller.filter_documents("engineer", docs)
    ids = {d["id"] for d in filtered}
    assert ids == {"d1", "d3"}  # Public + Internal only


# ======================================================================
# RateLimiter
# ======================================================================
def test_rate_limiter_allows_under_cap(storage):
    rl = RateLimiter(max_queries_per_minute=3, storage=storage)
    assert rl.is_allowed("alice") is True
    assert rl.is_allowed("alice") is True
    assert rl.is_allowed("alice") is True


def test_rate_limiter_blocks_over_cap(storage):
    rl = RateLimiter(max_queries_per_minute=2, storage=storage)
    assert rl.is_allowed("alice") is True
    assert rl.is_allowed("alice") is True
    assert rl.is_allowed("alice") is False


def test_rate_limiter_per_user_isolation(storage):
    rl = RateLimiter(max_queries_per_minute=1, storage=storage)
    assert rl.is_allowed("alice") is True
    assert rl.is_allowed("bob") is True
    assert rl.is_allowed("alice") is False
    assert rl.is_allowed("bob") is False


def test_rate_limiter_remaining(storage):
    rl = RateLimiter(max_queries_per_minute=3, storage=storage)
    assert rl.get_remaining_queries("alice") == 3
    rl.is_allowed("alice")
    assert rl.get_remaining_queries("alice") == 2
    rl.is_allowed("alice")
    rl.is_allowed("alice")
    assert rl.get_remaining_queries("alice") == 0


def test_rate_limiter_window_expiry(monkeypatch, storage):
    """Old timestamps drop out of the 60s sliding window."""
    rl = RateLimiter(max_queries_per_minute=2, storage=storage)
    base_time = [1000.0]
    monkeypatch.setattr(rl, "_now", lambda: base_time[0])
    assert rl.is_allowed("alice") is True
    assert rl.is_allowed("alice") is True
    assert rl.is_allowed("alice") is False  # at cap
    # Advance time past the window
    base_time[0] += 61
    assert rl.is_allowed("alice") is True  # window cleared


# ======================================================================
# CostEnforcer
# ======================================================================
def test_cost_enforcer_default_budgets():
    ce = CostEnforcer()
    assert ce.role_budgets["engineer"] == 100.0
    assert ce.role_budgets["executive"] == 1000.0


def test_cost_enforcer_allows_under_budget(storage):
    ce = CostEnforcer(storage=storage)
    assert ce.can_afford_query("alice", 10.0, role="engineer") is True


def test_cost_enforcer_blocks_over_budget(storage):
    ce = CostEnforcer(storage=storage)
    ce.add_cost("alice", "engineer", 95.0)
    assert ce.can_afford_query("alice", 10.0, role="engineer") is False  # 95+10>100
    assert ce.can_afford_query("alice", 4.0, role="engineer") is True  # 95+4<=100


def test_cost_enforcer_tracks_per_user(storage):
    ce = CostEnforcer(storage=storage)
    ce.add_cost("alice", "engineer", 50.0)
    ce.add_cost("bob", "engineer", 30.0)
    assert ce.get_user_spending("alice") == 50.0
    assert ce.get_user_spending("bob") == 30.0


def test_cost_enforcer_remaining_budget(storage):
    ce = CostEnforcer(storage=storage)
    ce.add_cost("alice", "manager", 100.0)
    # Manager budget is $500; spent $100; remaining $400
    assert ce.get_budget_remaining("alice", role="manager") == 400.0


def test_cost_enforcer_unknown_role_zero_budget(storage):
    ce = CostEnforcer(storage=storage)
    # Unknown role -> 0 budget; any cost rejects
    assert ce.can_afford_query("alice", 0.01, role="intruder") is False


def test_cost_enforcer_custom_budgets(storage):
    ce = CostEnforcer(role_budgets={"intern": 5.0}, storage=storage)
    assert ce.role_budgets["intern"] == 5.0
    assert ce.can_afford_query("alice", 3.0, role="intern") is True
    ce.add_cost("alice", "intern", 4.0)
    assert ce.can_afford_query("alice", 2.0, role="intern") is False  # 4+2>5
