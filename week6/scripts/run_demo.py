"""
Week 6 demo runner: exercises every guardrail end-to-end through the live
agent + Gemini Flash on paid tier.

Four scenarios:

  A. Role-based redaction
       Same query asked by different roles; engineer's salary should be
       redacted, hr's should not.
  B. Successful policy / approval queries
       Cross-section of normal questions that should answer cleanly with
       no guardrail interference.
  C. Rate limit
       Five quick queries from one user with max_queries_per_minute=3; the
       4th and 5th must be denied with "Rate limit exceeded".
  D. Cost budget
       Engineer budget set very low; we pre-charge the user so the next
       call gets blocked with "Budget exceeded".

Run from week6/:

    py -3.12 scripts/run_demo.py

Each scenario's queries are printed in the transcript and the full
structured run is saved to data/demo_run.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

WEEK6 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK6))

from access_control_starter import CostEnforcer, RateLimiter  # noqa: E402
from app_starter import Agent  # noqa: E402


def _print_query(label: str, result: dict, ans_max: int = 280) -> None:
    if "error" in result and result.get("answer") is None:
        print(f"  [{label}] blocked: {result['error']}")
        return
    ans = (result.get("answer") or "").strip()
    if len(ans) > ans_max:
        ans = ans[:ans_max] + "..."
    print(f"  [{label}] answer: {ans}")
    print(f"           tokens: {result.get('tokens_used')}  cost: ${result.get('cost'):.5f}")


def main() -> int:
    db = str(WEEK6 / "data" / "techcorp.db")
    policy = str(WEEK6 / "data" / "access_control.json")

    transcript = {"scenarios": {}}

    # ------------------------------------------------------------------
    # Scenario A — role-based redaction
    # ------------------------------------------------------------------
    print("=" * 72)
    print("SCENARIO A — role-based redaction")
    print("=" * 72)
    agent_a = Agent(db_path=db, access_policy_path=policy,
                    max_queries_per_minute=30)
    a_results = []
    for role in ("engineer", "hr", "manager"):
        q = "Find employee Brian Yang and tell me his salary."
        print(f"\nrole={role}  query: {q}")
        time.sleep(2)
        r = agent_a.query(q, user_id=f"{role}_demo", user_role=role)
        _print_query(role, r)
        a_results.append({"role": role, "query": q, **r})
    transcript["scenarios"]["A_redaction"] = a_results

    # ------------------------------------------------------------------
    # Scenario B — successful policy / approval queries
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SCENARIO B — clean answers across question types")
    print("=" * 72)
    agent_b = Agent(db_path=db, access_policy_path=policy,
                    max_queries_per_minute=30)
    b_queries = [
        ("engineer", "What is the travel policy at TechCorp?"),
        ("finance", "What is the manager-level expense approval limit?"),
        ("hr", "How many PTO days does TechCorp offer by default?"),
    ]
    b_results = []
    for role, q in b_queries:
        print(f"\nrole={role}  query: {q}")
        time.sleep(2)
        r = agent_b.query(q, user_id=f"{role}_demoB", user_role=role)
        _print_query(role, r)
        b_results.append({"role": role, "query": q, **r})
    transcript["scenarios"]["B_clean_answers"] = b_results

    # ------------------------------------------------------------------
    # Scenario C — rate limit (max 3 per minute, attempt 5)
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SCENARIO C — rate limit (max_queries_per_minute=3, 5 attempts)")
    print("=" * 72)
    agent_c = Agent(db_path=db, access_policy_path=policy,
                    max_queries_per_minute=3)
    c_results = []
    for i in range(1, 6):
        q = f"What is the expense approval limit for ic3?"
        print(f"\nattempt {i}/5")
        # No LLM call needed for the blocked attempts; this is a guardrail-
        # only test. We DO pay for the first 3 successful queries though.
        if i <= 3:
            time.sleep(1)
        r = agent_c.query(q, user_id="burst_user", user_role="engineer")
        _print_query(f"attempt {i}", r)
        c_results.append({"attempt": i, "query": q, **r})
    transcript["scenarios"]["C_rate_limit"] = c_results

    # ------------------------------------------------------------------
    # Scenario D — cost / budget enforcement
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SCENARIO D — budget enforcement (engineer budget pre-spent)")
    print("=" * 72)
    agent_d = Agent(db_path=db, access_policy_path=policy,
                    max_queries_per_minute=30)
    # Pre-spend the engineer's monthly $100 budget down to $0.005 remaining
    # so the next call's preflight estimate of $0.01 trips the budget gate.
    agent_d.cost_enforcer.add_cost("broke_user", "engineer", 99.995)
    remaining = agent_d.cost_enforcer.get_budget_remaining("broke_user")
    print(f"pre-spent: budget_remaining = ${remaining:.4f} "
          f"(below preflight estimate $0.01)")
    q = "Find any employee with the last name Smith."
    print(f"\nquery: {q}")
    r = agent_d.query(q, user_id="broke_user", user_role="engineer")
    _print_query("broke_user", r)
    d_results = [{"query": q, "budget_remaining_before": remaining, **r}]
    transcript["scenarios"]["D_budget"] = d_results

    # ------------------------------------------------------------------
    # Save + summary
    # ------------------------------------------------------------------
    out = WEEK6 / "data" / "demo_run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(transcript, indent=2, default=str))

    print("\n" + "=" * 72)
    print("DEMO COMPLETE")
    print("=" * 72)
    spent = (
        sum(r.get("cost", 0.0) for r in a_results)
        + sum(r.get("cost", 0.0) for r in b_results)
        + sum(r.get("cost", 0.0) for r in c_results)
        + sum(r.get("cost", 0.0) for r in d_results)
    )
    print(f"Total billed (Flash paid tier): ${spent:.5f}")
    print(f"Wrote {out.relative_to(WEEK6)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
