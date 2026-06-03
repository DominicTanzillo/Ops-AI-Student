"""Week 6 guardrails: AccessController + RateLimiter + CostEnforcer.

Three classes that gate the agent. All three persist their state through
the shared Storage abstraction (default InMemoryStorage; Firestore-ready)
so audit log, rate-limit history, and user spending survive across
classes within a single process.

Design points (locked in design session):
  - AccessController loads access_control.json:
      roles[role].permissions[capability] : bool
      sensitive_fields[field].visibility  : list[role]
      sensitive_fields[field].redact      : bool (default True)
    Two questions answerable: can role view a permission-keyed resource;
    can role view a named sensitive field (and is it auto-redacted).
  - RateLimiter: sliding 60-second window per user_id. Pop expired
    timestamps on every check.
  - CostEnforcer: per-role monthly budget. Per-user running sum.
    Spending accumulates against the user's role's budget.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from time import time
from typing import Any, Optional

from .storage import Storage, get_storage

log = logging.getLogger(__name__)

# Default per-role monthly budgets (USD). Locked in design session.
DEFAULT_ROLE_BUDGETS = {
    "engineer":  100.0,
    "manager":   500.0,
    "hr":        200.0,
    "finance":   500.0,
    "executive": 1000.0,
}

# Default redaction sentinel.
REDACTED = "[REDACTED]"


# ----------------------------------------------------------------------
# AccessController
# ----------------------------------------------------------------------
class AccessController:
    """Role-based access control with field redaction + audit log.

    policy schema (per week5/data/access_control.json):
      {
        "roles": {role: {permissions: {capability: bool, ...}, ...}, ...},
        "sensitive_fields": {field: {visibility: [role,...], redact: bool, ...}, ...}
      }
    """

    def __init__(
        self,
        access_policy_path: str | Path,
        storage: Optional[Storage] = None,
    ):
        with open(access_policy_path) as f:
            self.policy: dict[str, Any] = json.load(f)
        self.storage: Storage = storage or get_storage()
        # In-memory audit cache (mirrors the storage append). Useful for
        # tests that want a synchronous view without re-querying storage.
        self.audit_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Capability checks
    # ------------------------------------------------------------------
    def has_permission(self, role: str, capability: str) -> bool:
        """Check the role.permissions[capability] flag. Unknown role/cap -> False."""
        role_def = self.policy.get("roles", {}).get(role, {})
        return bool(role_def.get("permissions", {}).get(capability, False))

    def can_view_document(self, role: str, document: dict[str, Any]) -> bool:
        """Check if role may view this document based on its sensitivity tier.

        The included data uses string sensitivity tags (e.g. 'Public',
        'Internal', 'Confidential', 'Restricted'). Mapping to roles:
          Public        -> all roles
          Internal      -> engineer, manager, hr, finance, executive
          Confidential  -> manager, hr, finance, executive
          Restricted    -> finance, executive
        """
        sens = (document.get("sensitivity") or "Public").lower()
        tier_allows = {
            "public":       {"engineer", "manager", "hr", "finance", "executive"},
            "internal":     {"engineer", "manager", "hr", "finance", "executive"},
            "confidential": {"manager", "hr", "finance", "executive"},
            "restricted":   {"finance", "executive"},
        }
        allowed = role in tier_allows.get(sens, set())
        self.log_access(role, f"document:{document.get('id', '?')}",
                        allowed, field=None, sensitivity=sens)
        return allowed

    def can_view_field(self, role: str, field_name: str) -> bool:
        """Sensitive fields default to allowed; only fields explicitly listed
        in policy.sensitive_fields gate by role."""
        sf = self.policy.get("sensitive_fields", {}).get(field_name)
        if sf is None:
            return True  # not a sensitive field
        allowed = role in sf.get("visibility", [])
        self.log_access(role, f"field:{field_name}", allowed, field=field_name)
        return allowed

    # ------------------------------------------------------------------
    # Redaction
    # ------------------------------------------------------------------
    def redact_response(self, role: str, response: str) -> str:
        """Redact sensitive-field values from a response string.

        For each sensitive field the role is NOT allowed to see, replace
        any occurrence of `"field_name": <value>` or `field_name: <value>`
        (json-ish or yaml-ish) with `field_name: [REDACTED]`.

        This is a defense-in-depth pass for cases where the LLM included
        a sensitive field's value in its prose. Best-effort, not a security
        boundary on its own.
        """
        out = response
        for field_name, meta in self.policy.get("sensitive_fields", {}).items():
            if not meta.get("redact", True):
                continue
            if role in meta.get("visibility", []):
                continue
            # Match  "field": <value>  or  "field": "value"
            #   handles ints, floats, strings, scalar values.
            out = re.sub(
                rf'"{re.escape(field_name)}"\s*:\s*("(?:[^"\\]|\\.)*"|[^,}}\n]+)',
                f'"{field_name}": "{REDACTED}"',
                out,
            )
            # Also handle bare `field: value` (less common in JSON output)
            out = re.sub(
                rf'(^|\W){re.escape(field_name)}\s*:\s*([^\s,]+)',
                rf'\1{field_name}: {REDACTED}',
                out,
                flags=re.MULTILINE,
            )
        return out

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------
    def log_access(
        self,
        role: str,
        resource: str,
        allowed: bool,
        field: Optional[str] = None,
        sensitivity: Optional[str] = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "resource": resource,
            "allowed": bool(allowed),
            "field": field,
            "sensitivity": sensitivity,
        }
        self.audit_log.append(entry)
        self.storage.append("audit_log", entry)

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Return the in-memory audit log (mirrors storage append)."""
        return list(self.audit_log)

    # ------------------------------------------------------------------
    # Document filter
    # ------------------------------------------------------------------
    def filter_documents(
        self,
        role: str,
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [d for d in documents if self.can_view_document(role, d)]


# ----------------------------------------------------------------------
# RateLimiter
# ----------------------------------------------------------------------
class RateLimiter:
    """Sliding-60s-window queries-per-user rate limiter.

    State: per-user list of monotonic-clock timestamps in storage under
    `rate_limit_timestamps/{user_id}`. On each check, drop timestamps
    older than 60s and count remaining.
    """

    WINDOW_SECONDS: float = 60.0

    def __init__(
        self,
        max_queries_per_minute: int = 30,
        storage: Optional[Storage] = None,
    ):
        self.max_queries_per_minute = max_queries_per_minute
        self.storage = storage or get_storage()

    def _now(self) -> float:
        # injectable; tests monkeypatch via subclass or monkeypatch.setattr
        return time()

    def _get_timestamps(self, user_id: str) -> list[float]:
        return self.storage.get("rate_limit_timestamps", user_id) or []

    def _set_timestamps(self, user_id: str, ts: list[float]) -> None:
        self.storage.set("rate_limit_timestamps", user_id, ts)

    def _prune(self, user_id: str) -> list[float]:
        now = self._now()
        ts = [t for t in self._get_timestamps(user_id) if (now - t) < self.WINDOW_SECONDS]
        self._set_timestamps(user_id, ts)
        return ts

    def is_allowed(self, user_id: str) -> bool:
        """Returns True AND records a new timestamp if user is under cap."""
        ts = self._prune(user_id)
        if len(ts) >= self.max_queries_per_minute:
            return False
        ts.append(self._now())
        self._set_timestamps(user_id, ts)
        return True

    def get_remaining_queries(self, user_id: str) -> int:
        ts = self._prune(user_id)
        return max(0, self.max_queries_per_minute - len(ts))


# ----------------------------------------------------------------------
# CostEnforcer
# ----------------------------------------------------------------------
class CostEnforcer:
    """Per-role monthly budget. Per-user spending running sum.

    Spending lives in storage under `user_spending/{user_id}` as a dict:
      {"role": str, "total": float, "since": iso8601-string}
    """

    def __init__(
        self,
        role_budgets: Optional[dict[str, float]] = None,
        storage: Optional[Storage] = None,
        policy_path: Optional[str | Path] = None,
    ):
        if role_budgets is not None:
            self.role_budgets = dict(role_budgets)
        elif policy_path is not None and Path(policy_path).exists():
            with open(policy_path) as f:
                p = json.load(f)
            self.role_budgets = p.get("role_budgets", DEFAULT_ROLE_BUDGETS)
        else:
            self.role_budgets = dict(DEFAULT_ROLE_BUDGETS)
        self.storage = storage or get_storage()

    def _get_spending(self, user_id: str) -> dict[str, Any]:
        return self.storage.get("user_spending", user_id) or {
            "role": None,
            "total": 0.0,
            "since": datetime.now(timezone.utc).isoformat(),
        }

    def add_cost(self, user_id: str, role: str, cost: float) -> None:
        rec = self._get_spending(user_id)
        rec["role"] = role
        rec["total"] = float(rec.get("total", 0.0)) + float(cost)
        self.storage.set("user_spending", user_id, rec)

    def get_user_spending(self, user_id: str) -> float:
        return float(self._get_spending(user_id).get("total", 0.0))

    def get_budget_remaining(self, user_id: str, role: Optional[str] = None) -> float:
        rec = self._get_spending(user_id)
        effective_role = role or rec.get("role") or "engineer"
        budget = self.role_budgets.get(effective_role, 0.0)
        return max(0.0, budget - float(rec.get("total", 0.0)))

    def can_afford_query(
        self,
        user_id: str,
        estimated_cost: float,
        role: Optional[str] = None,
    ) -> bool:
        return estimated_cost <= self.get_budget_remaining(user_id, role=role)
