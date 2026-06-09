"""Week 7 evaluation: run a fixed set of queries through the agent +
optimization layer; produce a cost breakdown and optimization impact report.

Two modes:
  --mock    : use MockLLMClient (zero cost). Demonstrates the framework
              end-to-end without burning tokens. Pricing is computed from
              fake token counts so the breakdown columns still populate.
  (default) : call real Gemini. Requires GOOGLE_API_KEY in env. Will spend
              actual API credits (small, well within free tier for ~12
              queries).

The script wraps `app.cost_optimization.OptimizationStrategy` around the
Week 6 `Agent`:
  - Cache lookup BEFORE building messages -> exact-match short-circuit
  - Model routing BEFORE LLM call -> switch agent.model per-query
  - Cache write AFTER successful LLM response
  - Per-query cost recorded into CostAnalyzer for the breakdown report

Run from week7/:
  python scripts/evaluate_agent.py --mock                     # cost-free demo
  python scripts/evaluate_agent.py                            # uses real Gemini
  python scripts/evaluate_agent.py --queries scripts/test_queries.json
  python scripts/evaluate_agent.py --out data/eval_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
WEEK7 = REPO / "week7"
sys.path.insert(0, str(WEEK7))

try:
    from dotenv import load_dotenv
    load_dotenv(WEEK7 / ".env")
except ImportError:
    pass

from app.access_control import AccessController, CostEnforcer, RateLimiter
from app.agent import Agent, FunctionCall, LLMResponse
from app.cost_optimization import (
    CostAnalyzer,
    FeedbackLoop,
    MockEmbedder,
    OptimizationStrategy,
)
from app.storage import InMemoryStorage

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("eval")


# ----------------------------------------------------------------------
# Cost-free mock LLM that returns deterministic responses by category.
# Mirrors the MockLLMClient from week7/tests/conftest.py but with
# canned answers keyed by question, so the same query returns the same
# answer (which is what the cache expects).
# ----------------------------------------------------------------------
class _StaticMockLLM:
    """Returns a canned text answer for each query; cycles through fixed
    token counts so the cost breakdown has variation."""

    def __init__(self, rng_seed: int = 0):
        self._rng = random.Random(rng_seed)
        self._answers: dict[str, str] = {}

    def generate(self, messages, tools, model) -> LLMResponse:
        # Extract the question from the last user message
        question = ""
        for m in messages:
            if m.get("role") == "user":
                question = m.get("text", "")
        # Stable canned answer
        if question not in self._answers:
            self._answers[question] = f"Mock answer for: {question[:60]!r}"
        answer = self._answers[question]
        # Fake token counts (Pro = ~2x Flash to make Flash routing visible)
        if "pro" in model:
            in_tok = self._rng.randint(150, 400)
            out_tok = self._rng.randint(80, 200)
        else:
            in_tok = self._rng.randint(60, 150)
            out_tok = self._rng.randint(20, 80)
        return LLMResponse(
            text=answer, input_tokens=in_tok, output_tokens=out_tok,
        )


# ----------------------------------------------------------------------
# Build the agent + optimization stack
# ----------------------------------------------------------------------
def _build_stack(use_mock: bool, db_path: Path, policy_path: Path):
    storage = InMemoryStorage()
    rl = RateLimiter(max_queries_per_minute=60, storage=storage)
    ce = CostEnforcer(storage=storage)  # default per-role budgets
    ac = AccessController(policy_path, storage=storage)
    opt = OptimizationStrategy(storage=storage, embedder=MockEmbedder())
    cost_analyzer = CostAnalyzer(storage=storage)
    feedback = FeedbackLoop(storage=storage)

    if use_mock:
        llm = _StaticMockLLM()
    else:
        from app.agent import GeminiClient
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            log.error("GOOGLE_API_KEY not set. Use --mock for cost-free demo, "
                      "or set the key in .env first.")
            sys.exit(2)
        llm = GeminiClient(api_key=api_key)

    agent = Agent(
        tools=[],  # no tools for the cost-eval phase; full tool flow is in main.py
        storage=storage,
        llm_client=llm,
        access_controller=ac,
        rate_limiter=rl,
        cost_enforcer=ce,
        db_path=str(db_path) if db_path.exists() else None,
    )
    return agent, opt, cost_analyzer, feedback


# ----------------------------------------------------------------------
# Run one query: cache check -> model route -> agent.query -> record cost
# ----------------------------------------------------------------------
def _run_one(
    agent: Agent,
    opt: OptimizationStrategy,
    analyzer: CostAnalyzer,
    *,
    question: str,
    role: str,
    category: str,
) -> dict[str, Any]:
    # Cache check (exact + semantic)
    cache_hit, cached = opt.apply_caching(question)
    if cache_hit:
        return {
            "question": question, "role": role, "category": category,
            "cached": True, "model": "(cache)", "answer": cached,
            "input_tokens": 0, "output_tokens": 0, "cost": 0.0,
            "iterations": 0,
        }

    # Model routing
    chosen_model = opt.select_model_by_complexity(question)
    agent.model = chosen_model

    # Run the agent (its query() goes through guardrails too)
    result = agent.query(question, user_id=role + "_demo", user_role=role)

    # Cache the response
    opt.apply_caching(question, response=result.get("answer", ""))

    # Record per-component cost for the breakdown report
    analyzer.record_query({
        "query_text": question,
        "user_role": role,
        "model": chosen_model,
        "iterations": result.get("iterations", 0),
        "llm_cost": result.get("cost", 0.0),
        "tool_cost": 0.0,
        "retrieval_cost": 0.0,
        "error_cost": 0.0,
        "total_cost": result.get("cost", 0.0),
    })

    return {
        "question": question, "role": role, "category": category,
        "cached": False, "model": chosen_model,
        "answer": (result.get("answer") or "")[:200],
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
        "cost": result.get("cost", 0.0),
        "iterations": result.get("iterations", 0),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--queries", default="scripts/test_queries.json")
    p.add_argument("--out", default="data/eval_results.json")
    p.add_argument("--mock", action="store_true",
                   help="Use mock LLM (no API key needed, zero cost).")
    p.add_argument("--db", default="../week5/data/techcorp.db",
                   help="Path to techcorp.db (relative to week7/).")
    p.add_argument("--policy", default="../week6/data/access_control.json")
    args = p.parse_args()

    db_path = (WEEK7 / args.db).resolve() if not Path(args.db).is_absolute() else Path(args.db)
    policy_path = (WEEK7 / args.policy).resolve() if not Path(args.policy).is_absolute() else Path(args.policy)
    queries_path = (WEEK7 / args.queries).resolve() if not Path(args.queries).is_absolute() else Path(args.queries)
    out_path = (WEEK7 / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)

    # Fall back to week5's access_control.json if the week6 copy is missing
    if not policy_path.exists():
        alt = REPO / "week5" / "data" / "access_control.json"
        if alt.exists():
            policy_path = alt

    with open(queries_path) as f:
        queries = json.load(f)["test_queries"]

    log.info("=" * 76)
    log.info("WEEK 7 AGENT EVALUATION  (%d queries, mode=%s)",
             len(queries), "mock" if args.mock else "live")
    log.info("=" * 76)

    agent, opt, analyzer, feedback = _build_stack(
        use_mock=args.mock, db_path=db_path, policy_path=policy_path,
    )

    per_query: list[dict[str, Any]] = []
    for i, q in enumerate(queries, 1):
        r = _run_one(
            agent, opt, analyzer,
            question=q["question"], role=q.get("role", "engineer"),
            category=q.get("category", "uncategorized"),
        )
        per_query.append(r)
        log.info(
            "  [%2d/%d] %-70s  %-10s  cached=%-5s  cost=$%.5f",
            i, len(queries), q["question"][:70],
            r["model"].replace("gemini-", "")[:10],
            r["cached"], r["cost"],
        )

    # --- Reports
    breakdown = analyzer.get_cost_breakdown()
    impact = opt.get_optimization_impact()
    spikes = analyzer.identify_cost_spikes(sigma=2.0)

    log.info("\n" + "=" * 76)
    log.info("COST BREAKDOWN")
    log.info("=" * 76)
    for k, v in breakdown.items():
        if isinstance(v, float):
            log.info("  %-20s : $%.5f", k, v)
        else:
            log.info("  %-20s : %s", k, v)

    log.info("\n" + "=" * 76)
    log.info("OPTIMIZATION IMPACT")
    log.info("=" * 76)
    log.info("  cache_exact_hits         : %d", impact["cache_exact_hits"])
    log.info("  cache_semantic_hits      : %d", impact["cache_semantic_hits"])
    log.info("  cache_misses             : %d", impact["cache_misses"])
    log.info("  cache_hit_rate           : %.1f%%", 100 * impact["cache_hit_rate"])
    log.info("  model_choices            : %s", impact["model_choices"])
    log.info("  flash_share              : %.1f%%", 100 * impact["flash_share"])
    log.info("  responses_compressed     : %d", impact["responses_compressed"])

    if spikes:
        log.info("\n" + "=" * 76)
        log.info("COST SPIKES (>%dsigma)" % 2)
        log.info("=" * 76)
        for s in spikes:
            log.info("  $%.5f  %s", s["total_cost"], s.get("query_text", "?")[:70])

    # --- Demo: feedback submission
    log.info("\n" + "=" * 76)
    log.info("FEEDBACK LOOP DEMO")
    log.info("=" * 76)
    fb1 = feedback.submit_correction(
        original_query="What is the travel policy?",
        original_answer=per_query[0]["answer"][:120] if per_query else "",
        corrected_answer="See policy doc HR-001 for the current version.",
        user_role="manager",
    )
    fb2 = feedback.submit_correction(
        original_query="What is the PTO default?",
        original_answer=per_query[-1]["answer"][:120] if per_query else "",
        corrected_answer="Default 15 PTO days; 20 for managers.",
        user_role="engineer",
    )
    log.info("  manager submission   : %s", fb1)
    log.info("  engineer submission  : %s", fb2)
    log.info("  feedback metrics     : %s", feedback.get_feedback_metrics())

    # --- Write structured JSON
    report = {
        "mode": "mock" if args.mock else "live",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "per_query": per_query,
        "cost_breakdown": breakdown,
        "optimization_impact": impact,
        "spikes": spikes,
        "feedback_metrics": feedback.get_feedback_metrics(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log.info("\nwrote %s", out_path.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
