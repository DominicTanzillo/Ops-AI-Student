"""
Week 6: Agent with Access Control, Rate Limiting & Cost Enforcement

This is the Week 5 agent (Tool/EmployeeLookupTool/PolicySearchTool/
ExpenseQueryTool/Agent) copied into week6/ and updated to use the three
guardrails defined in access_control_starter.py:

    AccessController  - role-based view/redaction
    RateLimiter       - 30 queries / 60s per user_id
    CostEnforcer      - per-role monthly budgets

Run from the week6/ directory:

    python app_starter.py

If GOOGLE_API_KEY is set in the environment, real Gemini 2.5 Pro is used.
If not, the inline mock LLM is used so the test block still prints the
expected output without making any API calls or spending any money.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sqlite3
import sys
from typing import Any, Dict, List, Optional

from access_control_starter import AccessController, CostEnforcer, RateLimiter

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

GEMINI_INPUT_RATE_PER_1M = 0.075
GEMINI_OUTPUT_RATE_PER_1M = 0.30
GEMINI_MODEL = "gemini-2.5-pro"


# ============================================================================
# Tool base + 3 tools (from Week 5)
# ============================================================================


class Tool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def execute(self, **kwargs) -> str:
        raise NotImplementedError


class EmployeeLookupTool(Tool):
    def __init__(self, db_path: str):
        super().__init__(
            "employee_lookup",
            "Find employee information by name or ID",
        )
        self.db_path = db_path

    def execute(self, employee_name: Optional[str] = None,
                employee_id: Optional[str] = None) -> str:
        try:
            if not os.path.exists(self.db_path):
                return f"Error: database not found at {self.db_path}"
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            if employee_id is not None:
                cur.execute("SELECT * FROM employees WHERE id = ?", (employee_id,))
            elif employee_name:
                cur.execute(
                    "SELECT * FROM employees WHERE name LIKE ? LIMIT 10",
                    (f"%{employee_name}%",),
                )
            else:
                conn.close()
                return "Error: provide employee_name or employee_id"
            rows = [dict(r) for r in cur.fetchall()]
            conn.close()
            if not rows:
                return "Employee not found"
            return json.dumps(rows, default=str)
        except Exception as e:
            logger.error(f"Employee lookup error: {e}")
            return f"Error: {e}"


class PolicySearchTool(Tool):
    DOCS_PATH = os.path.join("data", "documents.json")

    def __init__(self):
        super().__init__(
            "policy_search",
            "Search policy documents by keyword or topic",
        )
        self.documents: List[Dict[str, Any]] = []
        if os.path.exists(self.DOCS_PATH):
            with open(self.DOCS_PATH, "r", encoding="utf-8") as f:
                self.documents = json.load(f)

    def execute(self, query: str, limit: int = 5) -> str:
        try:
            q = (query or "").lower().strip()
            if not q:
                return "Error: empty query"
            matches: List[Dict[str, Any]] = []
            for doc in self.documents:
                hay = (doc.get("content", "") + " " + doc.get("title", "")).lower()
                if q in hay:
                    matches.append(doc)
                if len(matches) >= limit:
                    break
            if not matches:
                return "No matching policy documents found"
            out = []
            for m in matches:
                snippet = (m.get("content", "") or "")[:500].replace("\n", " ")
                out.append(f"- {m.get('title', m.get('id', 'untitled'))}: {snippet}")
            return "\n".join(out)
        except Exception as e:
            logger.error(f"Policy search error: {e}")
            return f"Error: {e}"


class ExpenseQueryTool(Tool):
    POLICIES_PATH = os.path.join("data", "policies.json")

    def __init__(self):
        super().__init__(
            "expense_query",
            "Query expense approval limits by role",
        )
        self.policies: Dict[str, Any] = {}
        if os.path.exists(self.POLICIES_PATH):
            with open(self.POLICIES_PATH, "r", encoding="utf-8") as f:
                self.policies = json.load(f)

    def execute(self, role: str) -> str:
        try:
            limits = self.policies.get("expense", {}).get("approval_limits", {})
            if role in limits:
                return f"Approval limit for {role}: ${limits[role]}"
            return f"Role not found: {role}"
        except Exception as e:
            logger.error(f"Expense query error: {e}")
            return f"Error: {e}"


# ============================================================================
# Inline mock LLM (same as week5; lets the test block run with no API key)
# ============================================================================


class _MockGeminiClient:
    def __init__(self):
        self._rng = random.Random(7)
        self.models = self

    def generate_content(self, *, model: str, contents: List[Dict[str, Any]],
                         **_) -> "_MockResponse":
        text = self._answer_for(contents)
        in_tok = self._rng.randint(180, 350)
        out_tok = self._rng.randint(70, 200)
        return _MockResponse(text=text, input_tokens=in_tok, output_tokens=out_tok)

    @staticmethod
    def _answer_for(contents: List[Dict[str, Any]]) -> str:
        last_user = ""
        for c in contents:
            parts = c.get("parts", []) if isinstance(c, dict) else []
            for p in parts:
                if isinstance(p, dict) and "text" in p:
                    last_user = p["text"]
        q = last_user.lower()
        if "salary" in q:
            # Intentionally contains the literal "salary: $..." pattern so the
            # redaction path has something to redact when the test exercises it.
            return ("The requested employee record shows salary: $120,000 and "
                    "ssn: 123-45-6789. Manager approval recorded.")
        if "travel" in q:
            return ("TechCorp's travel policy requires pre-approval for all "
                    "business travel. Domestic flights up to $5000, hotels "
                    "tiered $150-$350, and per-diem meal limits apply.")
        if "expense" in q or "approval" in q:
            return ("Approval limits scale by role: IC1/IC2 $500, IC3 $2000, "
                    "Manager $5000, Director $25000, VP $100000.")
        return "Synthesized answer from the policy and tool results above."


class _MockResponse:
    def __init__(self, text: str, input_tokens: int, output_tokens: int):
        self.text = text
        self.usage_metadata = _Usage(input_tokens, output_tokens)


class _Usage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.prompt_token_count = input_tokens
        self.candidates_token_count = output_tokens
        self.total_token_count = input_tokens + output_tokens


# ============================================================================
# Agent (Week 5 logic + guardrails wired in for Week 6)
# ============================================================================


# Estimate for the budget pre-check at the start of each query. Roughly
# matches the cost of a single tool-using Gemini call (input + short output).
PREFLIGHT_COST_ESTIMATE = 0.01


class Agent:
    def __init__(self, db_path: str, api_key: Optional[str] = None,
                 access_policy_path: str = "data/access_control.json",
                 max_queries_per_minute: int = 30):
        self.db_path = db_path
        self.api_key = api_key or GOOGLE_API_KEY

        if self.api_key:
            try:
                import google.genai as genai  # noqa: WPS433
                self.client = genai.Client(api_key=self.api_key)
                self._mode = "live"
            except Exception as e:
                logger.warning(
                    "google-genai unavailable (%s); using mock LLM.", e,
                )
                self.client = _MockGeminiClient()
                self._mode = "mock"
        else:
            logger.info("GOOGLE_API_KEY not set; using mock LLM (no cost).")
            self.client = _MockGeminiClient()
            self._mode = "mock"

        self.tools: Dict[str, Tool] = {
            "employee_lookup": EmployeeLookupTool(db_path),
            "policy_search": PolicySearchTool(),
            "expense_query": ExpenseQueryTool(),
        }

        # ---- guardrails ---------------------------------------------------
        self.access_controller = AccessController(access_policy_path)
        self.rate_limiter = RateLimiter(max_queries_per_minute=max_queries_per_minute)
        self.cost_enforcer = CostEnforcer()

        self.token_count = 0
        self.total_cost = 0.0
        self.queries_run = 0

    # --- prompt + parsing -------------------------------------------------
    def _build_system_prompt(self, user_role: str) -> str:
        tool_lines = "\n".join(
            f"- {t.name}: {t.description}" for t in self.tools.values()
        )
        return (
            "You are a TechCorp assistant. Answer employee questions using "
            "the tools below.\n"
            f"User role: {user_role}\n\n"
            f"Available tools:\n{tool_lines}\n\n"
            "To use a tool, respond with:\n"
            "TOOL: <tool_name>\n"
            "ARGS: <argument>=<value>"
        )

    def _parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        m_name = re.search(r"TOOL:\s*([a-zA-Z_]+)", text)
        if not m_name:
            return None
        name = m_name.group(1).strip()
        if name not in self.tools:
            return None
        args: Dict[str, Any] = {}
        m_args = re.search(r"ARGS:\s*(.+)", text)
        if m_args:
            for pair in m_args.group(1).split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    args[k.strip()] = v.strip()
        return {"name": name, "args": args}

    def _estimate_query_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = (input_tokens / 1_000_000) * GEMINI_INPUT_RATE_PER_1M
        output_cost = (output_tokens / 1_000_000) * GEMINI_OUTPUT_RATE_PER_1M
        return input_cost + output_cost

    def _call_llm(self, system_prompt: str, user_text: str) -> _MockResponse:
        contents = [
            {"role": "user", "parts": [{"text": system_prompt}]},
            {"role": "user", "parts": [{"text": user_text}]},
        ]
        return self.client.models.generate_content(
            model=GEMINI_MODEL, contents=contents,
        )

    # --- guardrailed query ------------------------------------------------
    def query(self, user_query: str, user_id: str = "default_user",
              user_role: str = "engineer") -> Dict[str, Any]:
        # Guardrail 1: rate limit
        if not self.rate_limiter.is_allowed(user_id):
            return {
                "answer": None, "error": "Rate limit exceeded",
                "tokens_used": 0, "cost": 0.0, "role": user_role,
            }

        # Guardrail 2: budget pre-check
        if not self.cost_enforcer.can_afford_query(
            user_id, PREFLIGHT_COST_ESTIMATE, role=user_role,
        ):
            return {
                "answer": None, "error": "Budget exceeded",
                "tokens_used": 0, "cost": 0.0, "role": user_role,
            }

        logger.info("Processing query: %s", user_query)
        sys_prompt = self._build_system_prompt(user_role)

        total_in = 0
        total_out = 0

        first = self._call_llm(sys_prompt, user_query)
        total_in += first.usage_metadata.prompt_token_count
        total_out += first.usage_metadata.candidates_token_count

        tool_call = self._parse_tool_call(first.text)
        if tool_call:
            tool = self.tools[tool_call["name"]]
            try:
                tool_result = tool.execute(**tool_call["args"])
            except Exception as e:
                tool_result = f"Tool error: {e}"

            followup = self._call_llm(
                sys_prompt,
                f"User question: {user_query}\n\n"
                f"Tool {tool_call['name']} returned:\n{tool_result}\n\n"
                "Write a clear final answer.",
            )
            total_in += followup.usage_metadata.prompt_token_count
            total_out += followup.usage_metadata.candidates_token_count
            answer_text = followup.text
        else:
            answer_text = first.text

        # Guardrail 3: redact response by role
        answer_text = self.access_controller.redact_response(user_role, answer_text)

        cost = self._estimate_query_cost(total_in, total_out)

        # Guardrail 4: record actual cost
        self.cost_enforcer.add_cost(user_id, user_role, cost)

        self.token_count += total_in + total_out
        self.total_cost += cost
        self.queries_run += 1

        return {
            "answer": answer_text,
            "tokens_used": total_in + total_out,
            "cost": cost,
            "role": user_role,
        }

    def get_metrics(self) -> Dict[str, Any]:
        avg = (self.total_cost / self.queries_run) if self.queries_run else 0.0
        return {
            "total_queries": self.queries_run,
            "total_tokens": self.token_count,
            "total_cost": round(self.total_cost, 6),
            "avg_cost_per_query": round(avg, 6),
        }


# ============================================================================
# Test block
# ============================================================================


if __name__ == "__main__":
    try:
        agent = Agent("data/techcorp.db")
        print("Agent initialized successfully (with guardrails)")

        # Scenario 1: engineer asks about salary - mock LLM emits "salary: $..."
        # text; AccessController.redact_response should mask it for engineer.
        print("\n[1/3] engineer asks: 'What is the salary of employee Smith?'")
        r1 = agent.query(
            "What is the salary of employee Smith?",
            user_id="emp_001", user_role="engineer",
        )
        print(f"Answer (engineer, redacted): {r1.get('answer')}")
        assert r1.get("answer") and "[REDACTED]" in r1["answer"], \
            "Expected engineer's view to have salary redacted"
        print("  redaction: PASSED")

        # Scenario 2: HR asks the same; should NOT be redacted.
        print("\n[2/3] hr asks: 'What is the salary of employee Smith?'")
        r2 = agent.query(
            "What is the salary of employee Smith?",
            user_id="hr_001", user_role="hr",
        )
        print(f"Answer (hr, visible): {r2.get('answer')}")
        assert r2.get("answer") and "[REDACTED]" not in r2["answer"], \
            "Expected HR's view to NOT redact salary"
        print("  full visibility: PASSED")

        # Scenario 3: travel policy query (no redaction needed).
        print("\n[3/3] engineer asks: 'What is the travel policy?'")
        r3 = agent.query(
            "What is the travel policy?",
            user_id="emp_001", user_role="engineer",
        )
        print(f"Answer: {r3.get('answer')}")
        print(f"Tokens: {r3.get('tokens_used')}")
        print(f"Cost: ${r3.get('cost'):.6f}")

        metrics = agent.get_metrics()
        print(f"\nMetrics: {metrics}")
        print("\nAll guardrails working.")

    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Error during test")
        sys.exit(1)
