"""
Week 5 demo runner: 10 diverse queries through the live agent.

Reads GOOGLE_API_KEY from week5/.env (via python-dotenv inside app_starter.py)
and runs each query against Gemini. Sleeps between calls to stay inside the
free-tier rate limit. Retries once on transient 5xx responses. Captures
per-query results plus a final metrics dict, writes them to data/demo_run.json
and prints a screenshot-friendly transcript.

Run from week5/:

    py -3.12 scripts/run_demo.py

The expected output is ten Query / Answer / Tokens / Cost blocks followed by
a final Metrics dict, suitable for pasting / screenshotting into the Week 5
report.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

WEEK5 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK5))

from app_starter import Agent  # noqa: E402

QUERIES = [
    "What is the travel policy at TechCorp?",
    "Find employee Brian Yang.",
    "What's the expense approval limit for a manager?",
    "What's the expense approval limit for ic3?",
    "Look up the employee record for Edward Fuller.",
    "What's TechCorp's compensation review schedule?",
    "What's the expense approval limit for a VP?",
    "How many PTO days does TechCorp offer by default?",
    "What's the receipt threshold for expense submissions?",
    "Find any employee with the last name Smith.",
]


def _run_with_retry(agent: Agent, query: str, retries: int = 1) -> dict:
    """Conservative retry to protect against burning quota / credits.

    - Daily-cap 429s: do NOT retry. Sleeping doesn't help; we'd just keep
      bouncing off the same wall and double-bill on a paid tier.
    - Per-minute 429s: retry once after 45s.
    - Transient 5xx: retry once after 5s.
    - Anything else: fail fast.
    """
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return agent.query(query)
        except Exception as e:
            msg = str(e)
            last_err = e
            if "PerDay" in msg or "RequestsPerDay" in msg:
                raise  # daily cap hit; retry won't help
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                if attempt < retries:
                    time.sleep(45)
                    continue
                raise
            if "503" in msg or "UNAVAILABLE" in msg or "500" in msg:
                if attempt < retries:
                    time.sleep(5)
                    continue
                raise
            raise
    assert last_err is not None
    raise last_err


def main() -> int:
    agent = Agent(str(WEEK5 / "data" / "techcorp.db"))
    print("Agent initialized successfully")
    print(f"Model: {agent.client.__class__.__module__}.{agent.client.__class__.__name__}")
    print(f"Mode: {agent._mode}")
    print("=" * 70)

    results = []
    for i, q in enumerate(QUERIES, 1):
        print(f"\n[{i}/{len(QUERIES)}] Query: {q}")
        try:
            r = _run_with_retry(agent, q)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"query": q, "error": str(e)})
            time.sleep(7)
            continue
        ans = (r.get("answer") or "").strip()
        if len(ans) > 280:
            ans = ans[:280] + "..."
        print(f"  Answer: {ans}")
        print(f"  Tokens: {r.get('tokens_used')}")
        print(f"  Cost:   ${r.get('cost'):.6f}")
        results.append({
            "query": q,
            "answer": r.get("answer"),
            "tokens_used": r.get("tokens_used"),
            "cost": r.get("cost"),
        })
        # Paid-tier Flash has very high RPM; a short pause keeps the
        # transcript readable without burning wall-clock time.
        time.sleep(3)

    print("\n" + "=" * 70)
    metrics = agent.get_metrics()
    print(f"Final Metrics: {metrics}")

    out = WEEK5 / "data" / "demo_run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": "gemini-2.5-flash",
        "queries": results,
        "metrics": metrics,
    }, indent=2, default=str))
    print(f"\nWrote {out.relative_to(WEEK5)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
