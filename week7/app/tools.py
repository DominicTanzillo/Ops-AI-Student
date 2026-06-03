"""Agent tools for Week 5.

Three tools the agent can call via Gemini native function-calling:

  - EmployeeLookupTool : SQLite query against TechCorp's employees table.
                        Returns full row(s); does NOT redact (that is
                        Week 6's AccessController layer).
  - PolicySearchTool   : Keyword search over documents.json (74 policy
                        documents). Returns top-N with title + snippet.
  - ExpenseQueryTool   : Structured lookup over policies.json (approval
                        limits, per-diem rates, deadlines). Optional
                        role filter narrows approval-limit responses.

All three are zero-cost - no LLM, no external API. They can be unit-tested
against the real techcorp.db / json without any Gemini key.

Each Tool exposes a .to_function_declaration() that returns a dict
matching google.genai's FunctionDeclaration shape, so the agent can
register them with Gemini's native tool-calling API.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

WEEK5_DATA = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DB = WEEK5_DATA / "techcorp.db"
DEFAULT_DOCS = WEEK5_DATA / "documents.json"
DEFAULT_POLICIES = WEEK5_DATA / "policies.json"


class Tool:
    """Base class for agent tools.

    Subclasses set name + description in __init__ and override execute().
    parameters_schema is a JSON-Schema-style dict the agent uses to
    advertise the tool to Gemini.
    """

    name: str = ""
    description: str = ""
    parameters_schema: dict[str, Any] = {}

    def execute(self, **kwargs: Any) -> str:
        raise NotImplementedError

    def to_function_declaration(self) -> dict[str, Any]:
        """Return a dict that matches google.genai.types.FunctionDeclaration."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }


class EmployeeLookupTool(Tool):
    """Look up employee records from the TechCorp SQLite database.

    Returns matching row(s) as JSON. Does NOT redact sensitive fields.
    Redaction is the AccessController's job (Week 6) and happens at the
    agent's response-synthesis step.
    """

    name = "employee_lookup"
    description = (
        "Look up TechCorp employee records by full or partial name, or by "
        "employee ID. Returns up to 5 matches with all stored fields "
        "(id, name, email, department, title, salary, hire_date, manager). "
        "Use this when the user asks about a specific person or for "
        "directory lookups."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "employee_name": {
                "type": "string",
                "description": (
                    "Full or partial employee name (case-insensitive "
                    "substring match). Use this OR employee_id."
                ),
            },
            "employee_id": {
                "type": "integer",
                "description": "Exact employee ID. Use this OR employee_name.",
            },
            "limit": {
                "type": "integer",
                "description": "Max rows to return (default 5).",
            },
        },
    }

    def __init__(self, db_path: str | Path = DEFAULT_DB):
        self.db_path = str(db_path)

    def execute(
        self,
        employee_name: str | None = None,
        employee_id: int | None = None,
        limit: int = 5,
    ) -> str:
        if not employee_name and not employee_id:
            return json.dumps({
                "error": "must provide employee_name or employee_id",
            })
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            if employee_id is not None:
                cur.execute(
                    "SELECT * FROM employees WHERE id = ? LIMIT ?",
                    (employee_id, limit),
                )
            else:
                cur.execute(
                    "SELECT * FROM employees WHERE name LIKE ? LIMIT ?",
                    (f"%{employee_name}%", limit),
                )
            rows = [dict(r) for r in cur.fetchall()]
            conn.close()
            if not rows:
                return json.dumps({"matches": [], "note": "no employees found"})
            return json.dumps({"matches": rows, "count": len(rows)}, default=str)
        except Exception as e:
            log.exception("employee_lookup failed")
            return json.dumps({"error": str(e)})


class PolicySearchTool(Tool):
    """Keyword search across the 74 TechCorp policy documents.

    Loads documents.json once at construction. Scores by case-insensitive
    substring count in title (weight 3x) and content (weight 1x). Returns
    top-N matches with a content snippet.
    """

    name = "policy_search"
    description = (
        "Search TechCorp policy documents by keyword. Use this for questions "
        "about HR policies, engineering practices, security guidelines, "
        "benefits, compensation philosophy, and similar narrative policies. "
        "For numeric policy limits (approval thresholds, per diems, budgets) "
        "prefer expense_query."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search keywords (substring match on title + content).",
            },
            "category": {
                "type": "string",
                "description": (
                    "Optional category filter. Examples: 'HR', 'Engineering', "
                    "'Security', 'Finance'."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Max documents to return (default 5).",
            },
        },
        "required": ["query"],
    }

    def __init__(self, documents_path: str | Path = DEFAULT_DOCS):
        self.documents_path = Path(documents_path)
        with open(self.documents_path) as f:
            self.documents: list[dict[str, Any]] = json.load(f)

    def execute(
        self,
        query: str,
        category: str | None = None,
        limit: int = 5,
    ) -> str:
        q = (query or "").lower().strip()
        if not q:
            return json.dumps({"error": "query is required"})

        docs = self.documents
        if category:
            cat_lower = category.lower()
            docs = [d for d in docs if d.get("category", "").lower() == cat_lower]

        scored = []
        for d in docs:
            title = (d.get("title") or "").lower()
            content = (d.get("content") or "").lower()
            score = title.count(q) * 3 + content.count(q)
            if score > 0:
                scored.append((score, d))
        scored.sort(key=lambda kv: -kv[0])

        results = []
        for score, d in scored[:limit]:
            content = d.get("content", "")
            snippet_idx = content.lower().find(q)
            if snippet_idx >= 0:
                start = max(0, snippet_idx - 80)
                end = min(len(content), snippet_idx + 300)
                snippet = content[start:end].strip()
            else:
                snippet = content[:300]
            results.append({
                "id": d.get("id"),
                "title": d.get("title"),
                "category": d.get("category"),
                "sensitivity": d.get("sensitivity"),
                "match_score": score,
                "snippet": snippet,
            })

        return json.dumps({
            "query": query,
            "category_filter": category,
            "results": results,
            "match_count": len(scored),
        })


class ExpenseQueryTool(Tool):
    """Structured lookup over the TechCorp policies.json file.

    The policies file contains numeric thresholds (approval limits,
    per-diem rates, budget caps, submission deadlines). This tool returns
    them by topic. Distinct from PolicySearchTool which does narrative
    text search.
    """

    name = "expense_query"
    description = (
        "Look up TechCorp's structured expense/travel/PTO/compensation "
        "policy limits. Use this for numeric questions like 'who approves "
        "$5000 expenses', 'what is the meal per-diem', 'what is the PTO "
        "default'. Topics: 'approval_limits', 'travel_limits', 'per_diem', "
        "'submission_deadline', 'pto', 'compensation', 'all'."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": (
                    "Which policy section to return: approval_limits, "
                    "travel_limits, per_diem, submission_deadline, pto, "
                    "compensation, or all."
                ),
            },
            "role": {
                "type": "string",
                "description": (
                    "Optional role filter for approval_limits "
                    "(ic1_ic2, ic3, manager, director, vp)."
                ),
            },
        },
        "required": ["topic"],
    }

    def __init__(self, policies_path: str | Path = DEFAULT_POLICIES):
        self.policies_path = Path(policies_path)
        with open(self.policies_path) as f:
            self.policies: dict[str, Any] = json.load(f)

    def execute(self, topic: str, role: str | None = None) -> str:
        topic = (topic or "").lower().strip()
        out: dict[str, Any] = {}
        expense = self.policies.get("expense", {})
        travel = self.policies.get("travel", {})

        if topic == "all":
            return json.dumps(self.policies)

        if topic in ("approval_limits", "approval"):
            limits = expense.get("approval_limits", {})
            if role:
                rl = role.lower()
                out["role"] = role
                out["approval_limit"] = limits.get(rl)
            else:
                out["approval_limits"] = limits
            return json.dumps(out)

        if topic in ("travel_limits", "travel"):
            return json.dumps({"travel": travel})

        if topic in ("per_diem", "perdiem", "meals"):
            return json.dumps({
                "meal_breakfast": travel.get("budget_limits", {}).get("meal_breakfast"),
                "meal_lunch": travel.get("budget_limits", {}).get("meal_lunch"),
                "meal_dinner": travel.get("budget_limits", {}).get("meal_dinner"),
            })

        if topic in ("submission_deadline", "deadline"):
            return json.dumps({
                "submission_deadline_days": expense.get("submission_deadline_days"),
                "receipt_required_above": expense.get("receipt_required_above"),
            })

        if topic == "pto":
            return json.dumps({"pto": self.policies.get("pto", {})})

        if topic == "compensation":
            return json.dumps({"compensation": self.policies.get("compensation", {})})

        return json.dumps({
            "error": f"unknown topic: {topic!r}",
            "available_topics": [
                "approval_limits", "travel_limits", "per_diem",
                "submission_deadline", "pto", "compensation", "all",
            ],
        })


def default_tools(db_path: str | Path | None = None) -> list[Tool]:
    """Convenience: instantiate the 3 default tools with default data paths."""
    return [
        EmployeeLookupTool(db_path=db_path or DEFAULT_DB),
        PolicySearchTool(),
        ExpenseQueryTool(),
    ]
