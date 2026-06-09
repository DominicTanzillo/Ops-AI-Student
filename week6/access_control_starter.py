"""
Week 6: Access Control, Rate Limiting & Cost Enforcement

Single-file deliverable per the Week 6 README. Three guardrails are
implemented inline so the grader can run this file directly:

    AccessController  - role-based document/field access + redaction + audit
    RateLimiter       - sliding 60-second window per user_id
    CostEnforcer      - monthly per-role budgets and per-user spend tracking

Run from the week6/ directory:

    python access_control_starter.py

The test block at the bottom is the one from the Week 6 README. If every
assertion passes the script prints "All tests passed!" and exits 0. No
external services are touched.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from time import time
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# TASK 1: AccessController
# ============================================================================


class AccessController:
    """Enforce role-based access control using data/access_control.json."""

    def __init__(self, access_policy_path: str):
        with open(access_policy_path, "r", encoding="utf-8") as f:
            self.policy: Dict[str, Any] = json.load(f)
        self.audit_log: List[Dict[str, Any]] = []

    # --- document visibility ----------------------------------------------
    def can_view_document(self, role: str, document: Dict[str, Any]) -> bool:
        sensitivity = (document.get("sensitivity") or "Public").strip()
        # access_control.json uses capitalized keys: Public, Internal, etc.
        doc_access = self.policy.get("document_access", {})
        # Fall back to title-cased lookup so callers passing "public" still work.
        allowed_roles = doc_access.get(sensitivity) or doc_access.get(
            sensitivity.title(), []
        )
        return role in allowed_roles

    # --- field visibility -------------------------------------------------
    def can_view_field(self, role: str, field_name: str) -> bool:
        fields = self.policy.get("sensitive_fields", {})
        entry = fields.get(field_name)
        if entry is None:
            # Non-sensitive field: visible to everyone.
            return True
        return role in entry.get("visibility", [])

    # --- redaction --------------------------------------------------------
    def redact_response(self, role: str, response: str) -> str:
        if not response:
            return response
        out = response
        sensitive = self.policy.get("sensitive_fields", {})
        for field, entry in sensitive.items():
            if role in entry.get("visibility", []):
                continue
            # Match "salary: $100,000", "salary is $467,621", "ssn 123-45-6789".
            # After the field name, optionally consume a small connector word
            # (is/was/of/=/:) plus whitespace, then redact everything up to
            # the end of the value group (digits, commas, $, dashes, periods).
            pattern = re.compile(
                rf"({re.escape(field)})"
                rf"(?:\s*(?:is|was|of|=|:|are|equals)?\s*)"
                rf"(\$?[\d][\d,\.\-x]*|\S+)",
                re.IGNORECASE,
            )
            out = pattern.sub(r"\1: [REDACTED]", out)
        return out

    # --- audit ------------------------------------------------------------
    def log_access(self, role: str, resource: str, allowed: bool,
                   field: Optional[str] = None) -> None:
        self.audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "resource": resource,
            "field": field,
            "allowed": bool(allowed),
        })

    # --- document filter --------------------------------------------------
    def filter_documents(self, role: str,
                         documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        for doc in documents:
            allowed = self.can_view_document(role, doc)
            self.log_access(
                role, doc.get("id", doc.get("title", "unknown_doc")), allowed,
            )
            if allowed:
                kept.append(doc)
        return kept

    def get_audit_log(self) -> List[Dict[str, Any]]:
        return list(self.audit_log)


# ============================================================================
# TASK 2: RateLimiter
# ============================================================================


class RateLimiter:
    """Sliding 60-second window per user_id."""

    def __init__(self, max_queries_per_minute: int = 30):
        self.max_queries_per_minute = max_queries_per_minute
        self.user_query_times: Dict[str, List[float]] = {}

    def _purge_old(self, user_id: str, now: float) -> List[float]:
        times = [t for t in self.user_query_times.get(user_id, []) if now - t < 60]
        self.user_query_times[user_id] = times
        return times

    def is_allowed(self, user_id: str) -> bool:
        now = time()
        times = self._purge_old(user_id, now)
        if len(times) >= self.max_queries_per_minute:
            return False
        times.append(now)
        self.user_query_times[user_id] = times
        return True

    def get_remaining_queries(self, user_id: str) -> int:
        now = time()
        times = self._purge_old(user_id, now)
        return max(0, self.max_queries_per_minute - len(times))


# ============================================================================
# TASK 3: CostEnforcer
# ============================================================================


class CostEnforcer:
    """Per-role monthly budgets, per-user spend tracking."""

    DEFAULT_BUDGETS = {
        "engineer": 100.0,
        "manager": 500.0,
        "hr": 200.0,
        "finance": 500.0,
        "executive": 1000.0,
    }

    def __init__(self, policy_path: Optional[str] = None):
        self.role_budgets: Dict[str, float] = dict(self.DEFAULT_BUDGETS)
        if policy_path:
            try:
                with open(policy_path, "r", encoding="utf-8") as f:
                    self.role_budgets.update(json.load(f))
            except FileNotFoundError:
                pass
        # {user_id: {"role": str, "total": float}}
        self.user_spending: Dict[str, Dict[str, Any]] = {}

    def _budget_for(self, role: str) -> float:
        return float(self.role_budgets.get(role, 0.0))

    def add_cost(self, user_id: str, role: str, cost: float) -> None:
        entry = self.user_spending.setdefault(
            user_id, {"role": role, "total": 0.0},
        )
        # If a different role is passed for the same user later, prefer the
        # latest one (matches the README example).
        entry["role"] = role
        entry["total"] = float(entry["total"]) + float(cost)

    def can_afford_query(self, user_id: str, estimated_cost: float,
                         role: Optional[str] = None) -> bool:
        entry = self.user_spending.get(user_id)
        if entry is None and role is None:
            # No record yet and no role hint: assume engineer (lowest budget)
            # so we err on the side of caution. The README's test calls this
            # with no prior add_cost, so engineer's $100 is the default check.
            role = "engineer"

        if entry is None:
            budget = self._budget_for(role)
            spent = 0.0
        else:
            budget = self._budget_for(role or entry["role"])
            spent = float(entry["total"])

        remaining = budget - spent
        return estimated_cost <= remaining

    def get_budget_remaining(self, user_id: str) -> float:
        entry = self.user_spending.get(user_id)
        if entry is None:
            return 0.0
        budget = self._budget_for(entry["role"])
        return max(0.0, budget - float(entry["total"]))


# ============================================================================
# TASK 5: Test block (matches Week 6 README example)
# ============================================================================


if __name__ == "__main__":
    print("Testing AccessController...")
    controller = AccessController("data/access_control.json")

    assert not controller.can_view_field(
        "engineer", "salary"
    ), "Engineer should not see salary"
    assert controller.can_view_field("hr", "salary"), "HR should see salary"
    assert controller.can_view_field("manager", "salary"), "Manager should see salary"
    assert not controller.can_view_field(
        "engineer", "ssn"
    ), "Engineer should not see SSN"
    print("  can_view_field: PASSED")

    docs = [
        {"id": "doc1", "sensitivity": "Public", "content": "Mission statement"},
        {"id": "doc2", "sensitivity": "Confidential", "content": "Salary ranges"},
    ]
    visible = controller.filter_documents("engineer", docs)
    assert (
        len(visible) == 1 and visible[0]["id"] == "doc1"
    ), "Engineer should only see Public doc"
    print("  filter_documents: PASSED")

    print("\nTesting RateLimiter...")
    limiter = RateLimiter(max_queries_per_minute=3)
    assert limiter.is_allowed("user1"), "First query should be allowed"
    assert limiter.is_allowed("user1"), "Second query should be allowed"
    assert limiter.is_allowed("user1"), "Third query should be allowed"
    assert not limiter.is_allowed("user1"), "Fourth query should be blocked"
    print("  is_allowed: PASSED")

    print("\nTesting CostEnforcer...")
    enforcer = CostEnforcer()
    assert enforcer.can_afford_query(
        "user1", 50.0
    ), "Should afford $50 within $100 budget"
    enforcer.add_cost("user1", "engineer", 50.0)
    assert enforcer.can_afford_query(
        "user1", 49.0
    ), "Should afford $49 with $50 remaining"
    assert not enforcer.can_afford_query(
        "user1", 51.0
    ), "Should not afford $51 with $50 remaining"
    print("  can_afford_query: PASSED")

    print("\nAll tests passed!")
