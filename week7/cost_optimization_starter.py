"""Week 7: Cost Optimization & Feedback Loop -- self-contained deliverable.

Three classes:
  CostAnalyzer         - record per-query cost breakdown (retrieval/llm/
                         tool/error), roll up totals, flag statistical
                         outliers (mean + N*stdev) as cost spikes.
  OptimizationStrategy - hybrid cache (exact match first, optional
                         semantic similarity fallback), keyword-based
                         model router (Flash for simple, Flash for
                         complex on the current Gemini pricing), top-k
                         retrieval cap, response compression.
  FeedbackLoop         - collect + validate user corrections with a
                         role-based authority hierarchy (manager+ to
                         validate).

This file is intentionally self-contained (no internal imports). To run
the inline tests:
    cd week7
    python cost_optimization_starter.py
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# TASK 1: CostAnalyzer
# ============================================================================
class CostAnalyzer:
    """Record and analyze per-query cost breakdowns."""

    def __init__(self):
        self.query_history: List[Dict[str, Any]] = []

    def record_query(self, query: Dict[str, Any]) -> None:
        """Record a query with its cost components.

        Missing components default to 0.0. If `total_cost` is not supplied,
        it is computed as the sum of the four component costs.
        """
        entry = {
            "query_text": query.get("query_text", ""),
            "retrieval_cost": float(query.get("retrieval_cost", 0.0)),
            "llm_cost": float(query.get("llm_cost", 0.0)),
            "tool_cost": float(query.get("tool_cost", 0.0)),
            "error_cost": float(query.get("error_cost", 0.0)),
            "timestamp": query.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        }
        entry["total_cost"] = float(
            query.get("total_cost")
            if query.get("total_cost") is not None
            else entry["retrieval_cost"] + entry["llm_cost"]
            + entry["tool_cost"] + entry["error_cost"]
        )
        for k in ("user_id", "user_role", "model", "iterations"):
            if k in query:
                entry[k] = query[k]
        self.query_history.append(entry)

    def get_cost_breakdown(self) -> Dict[str, Any]:
        """Return totals across all recorded queries."""
        if not self.query_history:
            return {
                "retrieval_total": 0.0, "llm_total": 0.0,
                "tool_total": 0.0, "error_total": 0.0,
                "total_daily": 0.0, "query_count": 0,
            }
        return {
            "retrieval_total": sum(q["retrieval_cost"] for q in self.query_history),
            "llm_total": sum(q["llm_cost"] for q in self.query_history),
            "tool_total": sum(q["tool_cost"] for q in self.query_history),
            "error_total": sum(q["error_cost"] for q in self.query_history),
            "total_daily": sum(q["total_cost"] for q in self.query_history),
            "query_count": len(self.query_history),
        }

    def identify_cost_spikes(self, sigma: float = 2.0) -> List[Dict[str, Any]]:
        """Return queries whose total_cost exceeds mean + sigma * stdev.

        Needs at least 3 recorded queries for stdev to be meaningful.
        """
        if len(self.query_history) < 3:
            return []
        costs = [q["total_cost"] for q in self.query_history]
        mean = statistics.mean(costs)
        stdev = statistics.pstdev(costs)
        if stdev == 0:
            return []
        threshold = mean + sigma * stdev
        return [q for q in self.query_history if q["total_cost"] > threshold]


# ============================================================================
# Optional embedder for the semantic-cache fallback
# ============================================================================
class MockEmbedder:
    """Deterministic 16-dim embedder (sha256-hashed). Cost-free; used by
    the inline tests. Identical strings -> identical vector -> cosine 1.0.
    Different strings produce uncorrelated vectors (cosine ~ 0)."""

    def embed(self, text: str) -> List[float]:
        h = hashlib.sha256(text.lower().encode("utf-8")).digest()
        out = []
        for i in range(16):
            byte_pair = h[i * 2:i * 2 + 2]
            v = int.from_bytes(byte_pair, "big") / 32767.5 - 1.0
            out.append(v)
        norm = math.sqrt(sum(x * x for x in out)) or 1.0
        return [x / norm for x in out]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


# ============================================================================
# TASK 2: OptimizationStrategy
# ============================================================================
COMPLEX_KEYWORDS = {
    "compare", "analyze", "why", "how does", "explain", "summarize",
    "breakdown", "calculate", "trade-off", "tradeoff", "pros and cons",
    "step by step",
}


class OptimizationStrategy:
    """Caching + model routing + retrieval reduction + response compression."""

    def __init__(
        self,
        embedder: Optional[MockEmbedder] = None,
        similarity_threshold: float = 0.92,
        reduced_retrieval_limit: int = 3,
    ):
        self.cache: Dict[str, str] = {}
        self._cache_vectors: List[Dict[str, Any]] = []
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.reduced_retrieval_limit = reduced_retrieval_limit
        self.strategies_applied: List[str] = []
        # Per-strategy counters for the impact report
        self._exact_hits = 0
        self._semantic_hits = 0
        self._misses = 0
        self._compressions = 0
        self._compression_chars_saved = 0
        self._model_choices = {"complex": 0, "simple": 0}

    @staticmethod
    def _cache_key(query: str) -> str:
        return query.strip().lower()

    def apply_caching(self, query: str, response: Optional[str] = None) -> Tuple[bool, str]:
        """Hybrid cache lookup.

        - Exact-match first (cheap, deterministic).
        - On miss, if an embedder is configured, fall back to semantic
          similarity over previously-cached queries.
        - On true miss, if `response` is provided, store it and return
          (False, response). Otherwise return (False, "").
        """
        key = self._cache_key(query)
        if key in self.cache:
            self._exact_hits += 1
            if "cache" not in self.strategies_applied:
                self.strategies_applied.append("cache")
            return True, self.cache[key]

        if self.embedder is not None and self._cache_vectors:
            qvec = self.embedder.embed(query)
            best_sim = 0.0
            best_resp: Optional[str] = None
            for entry in self._cache_vectors:
                sim = _cosine_similarity(qvec, entry["vector"])
                if sim > best_sim:
                    best_sim = sim
                    best_resp = entry["response"]
            if best_sim >= self.similarity_threshold and best_resp is not None:
                self._semantic_hits += 1
                if "semantic_cache" not in self.strategies_applied:
                    self.strategies_applied.append("semantic_cache")
                return True, best_resp

        self._misses += 1
        if response is not None:
            self.cache[key] = response
            if self.embedder is not None:
                self._cache_vectors.append({
                    "query": query,
                    "vector": self.embedder.embed(query),
                    "response": response,
                })
        return False, response or ""

    def optimize_retrieval_count(self, num_docs: int) -> int:
        """Cap retrieval at reduced_retrieval_limit (default 3)."""
        if "retrieval_reduction" not in self.strategies_applied:
            self.strategies_applied.append("retrieval_reduction")
        return min(int(num_docs), self.reduced_retrieval_limit)

    def select_model_by_complexity(self, query: str) -> str:
        """Route by keyword complexity.

        Note: Gemini 2.5 Pro's free tier was removed by Google in Oct 2025,
        so we map complex queries to gemini-2.5-flash and simple lookups
        to gemini-2.5-flash-lite. The architecture (capable tier vs cheap
        tier) is unchanged from the original Pro-vs-Flash design.
        """
        q = (query or "").lower()
        is_complex = any(kw in q for kw in COMPLEX_KEYWORDS)
        if is_complex:
            self._model_choices["complex"] += 1
            if "model_routing" not in self.strategies_applied:
                self.strategies_applied.append("model_routing")
            return "gemini-2.5-flash"
        self._model_choices["simple"] += 1
        if "model_routing" not in self.strategies_applied:
            self.strategies_applied.append("model_routing")
        return "gemini-2.5-flash-lite"

    def enable_response_compression(self, response: str, max_chars: int = 1000) -> str:
        """Truncate over-long responses with an explicit marker.

        Deliberately not an LLM summarization step -- a second LLM call
        usually costs more tokens than the truncation saves.
        """
        if response is None:
            return ""
        if len(response) <= max_chars:
            return response
        original_len = len(response)
        compressed = (
            response[:max_chars]
            + f"\n[...compressed: {original_len - max_chars} chars truncated...]"
        )
        self._compressions += 1
        self._compression_chars_saved += original_len - max_chars
        if "response_compression" not in self.strategies_applied:
            self.strategies_applied.append("response_compression")
        return compressed

    def get_optimization_impact(self) -> Dict[str, Any]:
        total_queries = self._exact_hits + self._semantic_hits + self._misses
        hit_rate = (
            (self._exact_hits + self._semantic_hits) / total_queries
            if total_queries > 0 else 0.0
        )
        total_model_calls = self._model_choices["complex"] + self._model_choices["simple"]
        simple_share = (
            self._model_choices["simple"] / total_model_calls
            if total_model_calls > 0 else 0.0
        )
        return {
            "total_savings_pct": round(100 * hit_rate, 1),
            "strategies_applied": list(self.strategies_applied),
            "breakdown": {
                "cache_exact_hits": self._exact_hits,
                "cache_semantic_hits": self._semantic_hits,
                "cache_misses": self._misses,
                "cache_hit_rate": hit_rate,
                "model_choices": dict(self._model_choices),
                "cheap_model_share": simple_share,
                "responses_compressed": self._compressions,
                "compression_chars_saved": self._compression_chars_saved,
            },
        }


# ============================================================================
# TASK 3: FeedbackLoop
# ============================================================================
class FeedbackLoop:
    """Collect and validate user corrections.

    Authority hierarchy (matches upstream README):
        engineer  = 1
        hr        = 2
        finance   = 2
        manager   = 3
        executive = 4
    A correction is accepted at submit time only if the submitter is
    manager+ (level >= 3) AND the corrected answer is more detailed than
    the original. validate_correction(index) re-applies the same checks
    against a stored entry.
    """

    MIN_AUTHORITY = 3

    def __init__(self):
        self.corrections: List[Dict[str, Any]] = []
        self.authority = {
            "engineer": 1,
            "hr": 2,
            "finance": 2,
            "manager": 3,
            "executive": 4,
        }

    def submit_correction(
        self,
        original_query: str,
        original_answer: str,
        corrected_answer: str,
        user_role: str,
    ) -> Dict[str, Any]:
        level = self.authority.get(user_role, 0)
        if not corrected_answer.strip():
            reason = "empty correction"
            accepted = False
        elif level < self.MIN_AUTHORITY:
            reason = f"role '{user_role}' lacks authority (need manager+)"
            accepted = False
        elif len(corrected_answer.strip()) <= len(original_answer.strip()):
            reason = "correction not more detailed than original"
            accepted = False
        else:
            reason = "ok"
            accepted = True

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_role": user_role,
            "original_query": original_query,
            "original_answer": original_answer,
            "corrected_answer": corrected_answer,
            "accepted": accepted,
            "reason": reason,
        }
        self.corrections.append(entry)
        return {"accepted": accepted, "reason": reason}

    def validate_correction(self, index: int) -> bool:
        if not (0 <= index < len(self.corrections)):
            return False
        c = self.corrections[index]
        level = self.authority.get(c.get("user_role", ""), 0)
        if level < self.MIN_AUTHORITY:
            return False
        if len(c["corrected_answer"].strip()) <= len(c["original_answer"].strip()):
            return False
        return True

    def get_feedback_metrics(self) -> Dict[str, Any]:
        if not self.corrections:
            return {
                "total_corrections": 0,
                "validation_rate": 0.0,
                "avg_correction_length": 0.0,
                "top_error_patterns": [],
            }
        accepted = [c for c in self.corrections if c.get("accepted")]
        validation_rate = len(accepted) / len(self.corrections)
        avg_len = sum(len(c["corrected_answer"]) for c in self.corrections) / len(self.corrections)
        # Top error patterns = the first words of original_query, counted
        from collections import Counter
        prefixes = Counter()
        for c in self.corrections:
            words = c["original_query"].lower().split()[:3]
            if words:
                prefixes[" ".join(words)] += 1
        top_patterns = [
            {"pattern": p, "count": n} for p, n in prefixes.most_common(3)
        ]
        return {
            "total_corrections": len(self.corrections),
            "validation_rate": round(validation_rate, 3),
            "avg_correction_length": round(avg_len, 1),
            "top_error_patterns": top_patterns,
        }


# ============================================================================
# Inline test block (run: python cost_optimization_starter.py)
# ============================================================================
if __name__ == "__main__":
    failures: List[str] = []

    def check(label: str, cond: bool, detail: str = "") -> None:
        status = "PASSED" if cond else f"FAILED ({detail})"
        print(f"  {label}: {status}")
        if not cond:
            failures.append(label)

    # ------------------------------------------------------------------
    # CostAnalyzer
    # ------------------------------------------------------------------
    print("Testing CostAnalyzer...")
    analyzer = CostAnalyzer()
    for i in range(10):
        analyzer.record_query({
            "query_text": f"baseline query {i}",
            "retrieval_cost": 0.001,
            "llm_cost": 0.005,
            "tool_cost": 0.0,
            "error_cost": 0.0,
            "total_cost": 0.006,
        })
    analyzer.record_query({
        "query_text": "expensive outlier",
        "retrieval_cost": 0.05,
        "llm_cost": 0.4,
        "tool_cost": 0.1,
        "error_cost": 0.0,
        "total_cost": 0.55,
    })
    breakdown = analyzer.get_cost_breakdown()
    check(
        "record_query",
        breakdown["query_count"] == 11,
        f"got {breakdown['query_count']}",
    )
    check(
        "get_cost_breakdown",
        abs(breakdown["total_daily"] - (10 * 0.006 + 0.55)) < 1e-9,
        f"got total_daily={breakdown['total_daily']:.4f}",
    )
    spikes = analyzer.identify_cost_spikes(sigma=2.0)
    check(
        "identify_cost_spikes",
        len(spikes) == 1 and spikes[0]["query_text"] == "expensive outlier",
        f"got {len(spikes)} spikes",
    )

    # ------------------------------------------------------------------
    # OptimizationStrategy
    # ------------------------------------------------------------------
    print("\nTesting OptimizationStrategy...")
    opt = OptimizationStrategy(embedder=MockEmbedder())

    hit1, _ = opt.apply_caching("What is the travel policy?", "See HR-001.")
    hit2, resp2 = opt.apply_caching("What is the travel policy?", "ignored second store")
    check(
        "apply_caching",
        hit1 is False and hit2 is True and resp2 == "See HR-001.",
        f"hit1={hit1} hit2={hit2} resp2={resp2!r}",
    )

    check(
        "optimize_retrieval_count",
        opt.optimize_retrieval_count(15) == 3 and opt.optimize_retrieval_count(2) == 2,
        f"got {opt.optimize_retrieval_count(15)}, {opt.optimize_retrieval_count(2)}",
    )

    m_simple = opt.select_model_by_complexity("What is the PTO policy?")
    m_complex = opt.select_model_by_complexity("Analyze why expenses spiked last quarter")
    check(
        "select_model_by_complexity",
        m_simple.endswith("flash-lite") and m_complex == "gemini-2.5-flash",
        f"simple={m_simple} complex={m_complex}",
    )

    long_resp = "x" * 1500
    short_resp = "x" * 100
    compressed = opt.enable_response_compression(long_resp)
    uncompressed = opt.enable_response_compression(short_resp)
    check(
        "enable_response_compression",
        len(compressed) < len(long_resp) and uncompressed == short_resp,
        f"compressed_len={len(compressed)}",
    )

    impact = opt.get_optimization_impact()
    check(
        "get_optimization_impact",
        "cache" in impact["strategies_applied"] and impact["breakdown"]["cache_exact_hits"] >= 1,
        f"strategies={impact['strategies_applied']}",
    )

    # ------------------------------------------------------------------
    # FeedbackLoop
    # ------------------------------------------------------------------
    print("\nTesting FeedbackLoop...")
    fb = FeedbackLoop()

    r_mgr = fb.submit_correction(
        original_query="What is the travel policy for flights over 8 hours?",
        original_answer="There is no specific policy for 8+ hour flights.",
        corrected_answer="Employees can book business class for flights over 8 hours with manager approval.",
        user_role="manager",
    )
    r_eng = fb.submit_correction(
        original_query="What is the PTO default?",
        original_answer="Default is 15 PTO days.",
        corrected_answer="Default 15 PTO days; managers get 20.",
        user_role="engineer",
    )
    r_short = fb.submit_correction(
        original_query="What is the expense limit?",
        original_answer="$2000 per trip for engineers.",
        corrected_answer="$2000.",
        user_role="executive",
    )
    check(
        "submit_correction",
        r_mgr["accepted"] is True
        and r_eng["accepted"] is False
        and r_short["accepted"] is False,
        f"mgr={r_mgr} eng={r_eng} short={r_short}",
    )

    check(
        "validate_correction",
        fb.validate_correction(0) is True
        and fb.validate_correction(1) is False
        and fb.validate_correction(99) is False,
        "validation results unexpected",
    )

    metrics = fb.get_feedback_metrics()
    check(
        "get_feedback_metrics",
        metrics["total_corrections"] == 3
        and 0.0 < metrics["validation_rate"] < 1.0
        and metrics["avg_correction_length"] > 0,
        f"metrics={metrics}",
    )

    # ------------------------------------------------------------------
    print()
    if failures:
        print(f"{len(failures)} test(s) failed: {failures}")
        raise SystemExit(1)
    print("All tests passed!")
