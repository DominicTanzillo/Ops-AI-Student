"""Week 7: Cost optimization + feedback loop.

Three classes (locked in Stage-2 design):

  CostAnalyzer        - records per-query cost breakdown (retrieval/llm/
                        tool/error), rolls them up, flags statistical
                        outliers as cost spikes.
  OptimizationStrategy - hybrid cache (exact-match first, semantic-
                        similarity fallback via a swappable Embedder),
                        keyword-based model router (Pro vs Flash),
                        retrieval count reduction, response compression.
  FeedbackLoop        - collects user corrections with role-based
                        validation (managers/hr/executive auto-validated,
                        engineers require a manager review step).

The semantic-similarity branch of the cache needs an Embedder. The default
GeminiEmbedder calls Gemini's text-embedding API (costs); a MockEmbedder
is provided for tests so the suite stays at $0.
"""
from __future__ import annotations

import json
import logging
import math
import os
import statistics
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Protocol

from .storage import Storage, get_storage

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Pricing (kept in sync with agent.py PRICING_PER_1M)
# ----------------------------------------------------------------------
EMBEDDING_PRICE_PER_1M = 0.025  # text-embedding-004 per 1M input tokens

# Tuning knobs (could move to env)
SEMANTIC_SIMILARITY_THRESHOLD = 0.92  # cosine sim above this == cache hit
DEFAULT_RETRIEVAL_LIMIT_REDUCED = 3

# Keyword sets used by select_model_by_complexity.
# Pro for analytical / multi-step / "why" questions.
# Flash for simple lookups.
COMPLEX_KEYWORDS = {
    "compare", "analyze", "why", "how does", "explain", "summarize",
    "breakdown", "calculate", "what's the difference", "step by step",
    "trade-off", "tradeoff", "pros and cons", "vs",
}


# ============================================================================
# CostAnalyzer
# ============================================================================
class CostAnalyzer:
    """Records + analyzes per-query cost breakdowns.

    Each recorded query is a dict with these fields (any missing is treated
    as 0.0):
      query_text, retrieval_cost, llm_cost, tool_cost, error_cost,
      total_cost, timestamp
    """

    def __init__(self, storage: Optional[Storage] = None):
        self.storage: Storage = storage or get_storage()

    def record_query(self, query: dict[str, Any]) -> None:
        entry = {
            "query_text": query.get("query_text", ""),
            "retrieval_cost": float(query.get("retrieval_cost", 0.0)),
            "llm_cost": float(query.get("llm_cost", 0.0)),
            "tool_cost": float(query.get("tool_cost", 0.0)),
            "error_cost": float(query.get("error_cost", 0.0)),
            "timestamp": query.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        }
        entry["total_cost"] = (
            entry["retrieval_cost"] + entry["llm_cost"]
            + entry["tool_cost"] + entry["error_cost"]
        )
        # Preserve caller-supplied total if explicitly set
        if "total_cost" in query and query["total_cost"] is not None:
            entry["total_cost"] = float(query["total_cost"])
        # Optional fields for diagnosis
        for k in ("user_id", "user_role", "model", "iterations"):
            if k in query:
                entry[k] = query[k]
        self.storage.append("query_history_costs", entry)

    def get_cost_breakdown(self) -> dict[str, Any]:
        entries = self.storage.query("query_history_costs")
        if not entries:
            return {
                "retrieval_total": 0.0, "llm_total": 0.0,
                "tool_total": 0.0, "error_total": 0.0,
                "total_daily": 0.0, "query_count": 0,
            }
        return {
            "retrieval_total": sum(e.get("retrieval_cost", 0.0) for e in entries),
            "llm_total": sum(e.get("llm_cost", 0.0) for e in entries),
            "tool_total": sum(e.get("tool_cost", 0.0) for e in entries),
            "error_total": sum(e.get("error_cost", 0.0) for e in entries),
            "total_daily": sum(e.get("total_cost", 0.0) for e in entries),
            "query_count": len(entries),
        }

    def identify_cost_spikes(self, sigma: float = 2.0) -> list[dict[str, Any]]:
        """Return queries with total_cost > mean + sigma * stdev.

        Requires at least 3 recorded queries (otherwise stdev is undefined
        or unstable).
        """
        entries = self.storage.query("query_history_costs")
        if len(entries) < 3:
            return []
        costs = [e.get("total_cost", 0.0) for e in entries]
        mean = statistics.mean(costs)
        stdev = statistics.pstdev(costs)
        if stdev == 0:
            return []
        threshold = mean + sigma * stdev
        return [e for e in entries if e.get("total_cost", 0.0) > threshold]


# ============================================================================
# Embedder protocol (swappable for cost-free testing)
# ============================================================================
class Embedder(Protocol):
    """Anything that turns a string into a fixed-length vector."""

    def embed(self, text: str) -> list[float]:
        ...


class MockEmbedder:
    """Deterministic test embedder.

    Hashes the input string into a fixed 16-dim float vector. Two strings
    produce the same vector iff they are identical. Cosine similarity of
    different strings is generally low; identical strings == 1.0.
    """

    def embed(self, text: str) -> list[float]:
        import hashlib
        h = hashlib.sha256(text.lower().encode("utf-8")).digest()
        # Convert 32 bytes into 16 floats in [-1, 1]
        out = []
        for i in range(16):
            byte_pair = h[i * 2:i * 2 + 2]
            v = int.from_bytes(byte_pair, "big") / 32767.5 - 1.0
            out.append(v)
        # Normalize so cosine ratios are stable
        norm = math.sqrt(sum(x * x for x in out)) or 1.0
        return [x / norm for x in out]


class GeminiEmbedder:
    """Live Gemini text-embedding wrapper. Lazy-imports the SDK."""

    def __init__(self, api_key: str, model: str = "text-embedding-004"):
        try:
            from google import genai  # noqa: WPS433
        except ImportError as e:
            raise RuntimeError(
                "google-genai required for GeminiEmbedder; "
                "use MockEmbedder for tests.",
            ) from e
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for GeminiEmbedder.")
        self._client = genai.Client(api_key=api_key)
        self.model = model

    def embed(self, text: str) -> list[float]:
        resp = self._client.models.embed_content(model=self.model, contents=text)
        return list(resp.embeddings[0].values)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


# ============================================================================
# OptimizationStrategy
# ============================================================================
class OptimizationStrategy:
    """Caching + model routing + retrieval reduction + response compression."""

    def __init__(
        self,
        storage: Optional[Storage] = None,
        embedder: Optional[Embedder] = None,
        similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
        reduced_retrieval_limit: int = DEFAULT_RETRIEVAL_LIMIT_REDUCED,
    ):
        self.storage: Storage = storage or get_storage()
        self.embedder: Optional[Embedder] = embedder
        self.similarity_threshold = similarity_threshold
        self.reduced_retrieval_limit = reduced_retrieval_limit
        # Track impact for the writeup
        self._exact_hits = 0
        self._semantic_hits = 0
        self._misses = 0
        self._compressions = 0
        self._compression_chars_saved = 0
        self._model_choices: dict[str, int] = {"pro": 0, "flash": 0}

    # ------------------------------------------------------------------
    # Caching - hybrid: exact-match first, then semantic fallback
    # ------------------------------------------------------------------
    def apply_caching(self, query: str, response: Optional[str] = None) -> tuple[bool, str]:
        """Look up (and optionally insert) a cached response for a query.

        Returns (is_hit, response):
          - is_hit=True: cache had a match; returned response is the cached one
          - is_hit=False: no match; if `response` was provided, it is cached
                          for next time; the returned string is `response` or ""

        Exact match check first (cheap, deterministic). On miss, if an
        embedder is configured, fall back to semantic similarity over all
        previously-cached queries.
        """
        key = self._cache_key(query)
        cached = self.storage.get("query_cache", key)
        if cached is not None:
            self._exact_hits += 1
            return True, cached["response"]

        # Semantic fallback
        if self.embedder is not None:
            qvec = self.embedder.embed(query)
            all_cached = self.storage.query("query_cache_index")  # vectors live here
            best_sim = 0.0
            best_resp: Optional[str] = None
            for entry in all_cached:
                sim = _cosine_similarity(qvec, entry["vector"])
                if sim > best_sim:
                    best_sim = sim
                    best_resp = entry["response"]
            if best_sim >= self.similarity_threshold and best_resp is not None:
                self._semantic_hits += 1
                return True, best_resp

        self._misses += 1
        if response is not None:
            self._insert_cache(query, response, key)
        return False, response or ""

    def _insert_cache(self, query: str, response: str, key: str) -> None:
        self.storage.set("query_cache", key, {"query": query, "response": response})
        if self.embedder is not None:
            self.storage.append("query_cache_index", {
                "query": query,
                "vector": self.embedder.embed(query),
                "response": response,
            })

    @staticmethod
    def _cache_key(query: str) -> str:
        return query.strip().lower()

    # ------------------------------------------------------------------
    # Retrieval count optimization
    # ------------------------------------------------------------------
    def optimize_retrieval_count(self, num_docs: int) -> int:
        """Cap retrieval at self.reduced_retrieval_limit."""
        return min(int(num_docs), self.reduced_retrieval_limit)

    # ------------------------------------------------------------------
    # Model selection - keyword-based router (locked in design)
    # ------------------------------------------------------------------
    def select_model_by_complexity(self, query: str) -> str:
        q = (query or "").lower()
        is_complex = any(kw in q for kw in COMPLEX_KEYWORDS)
        # Pro free tier was deprecated by Google (Oct 2025). For this run we
        # use Flash as the "expensive" tier and Flash-Lite as the cheap tier.
        # The architecture (cheap path for simple queries, capable path for
        # complex ones) is unchanged.
        choice = "gemini-2.5-flash" if is_complex else "gemini-2.5-flash-lite"
        self._model_choices["pro" if is_complex else "flash"] += 1
        return choice

    # ------------------------------------------------------------------
    # Response compression
    # ------------------------------------------------------------------
    def enable_response_compression(self, response: str, max_chars: int = 1000) -> str:
        """If a response is over max_chars, truncate with an explicit marker.

        Cheaper than re-running an LLM summarization (which would cost more
        tokens than it saves). This is a coarse compression - good enough to
        cut the worst outliers, simple enough to explain.
        """
        if response is None:
            return ""
        if len(response) <= max_chars:
            return response
        original_len = len(response)
        compressed = response[:max_chars] + f"\n[...compressed: {original_len - max_chars} chars truncated...]"
        self._compressions += 1
        self._compression_chars_saved += original_len - max_chars
        return compressed

    # ------------------------------------------------------------------
    # Impact reporting
    # ------------------------------------------------------------------
    def get_optimization_impact(self) -> dict[str, Any]:
        total_queries = self._exact_hits + self._semantic_hits + self._misses
        hit_rate = (
            (self._exact_hits + self._semantic_hits) / total_queries
            if total_queries > 0 else 0.0
        )
        flash_share = (
            self._model_choices["flash"]
            / max(1, self._model_choices["flash"] + self._model_choices["pro"])
        )
        return {
            "total_queries_seen": total_queries,
            "cache_exact_hits": self._exact_hits,
            "cache_semantic_hits": self._semantic_hits,
            "cache_misses": self._misses,
            "cache_hit_rate": hit_rate,
            "model_choices": dict(self._model_choices),
            "flash_share": flash_share,
            "responses_compressed": self._compressions,
            "compression_chars_saved": self._compression_chars_saved,
        }


# ============================================================================
# FeedbackLoop
# ============================================================================
# Roles whose corrections are auto-validated (we trust their judgment for the
# scope of an internal TechCorp agent).
AUTO_VALIDATE_ROLES = {"manager", "hr", "finance", "executive"}


class FeedbackLoop:
    """Collect + validate user corrections.

    Corrections from auto-validated roles are stored as validated=True
    immediately. Engineer corrections start as validated=False and require
    a separate validate_correction(index) call (a manager review step).
    """

    def __init__(self, storage: Optional[Storage] = None):
        self.storage: Storage = storage or get_storage()

    def submit_correction(
        self,
        original_query: str,
        original_answer: str,
        corrected_answer: str,
        user_role: str,
    ) -> dict[str, Any]:
        if not corrected_answer.strip():
            return {"accepted": False, "reason": "empty correction"}
        if corrected_answer.strip() == original_answer.strip():
            return {"accepted": False, "reason": "correction identical to original"}

        validated = user_role in AUTO_VALIDATE_ROLES
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_role": user_role,
            "original_query": original_query,
            "original_answer": original_answer,
            "corrected_answer": corrected_answer,
            "validated": validated,
        }
        self.storage.append("feedback_corrections", entry)
        return {"accepted": True, "validated": validated}

    def validate_correction(self, index: int) -> bool:
        """Approve a previously-submitted unvalidated correction (manager step)."""
        entries = self.storage.query("feedback_corrections")
        if not (0 <= index < len(entries)):
            return False
        # Storage layer doesn't expose update-by-index; we re-build the list.
        # Find the entry, mark validated, write back via clear+re-append.
        entries[index]["validated"] = True
        # Replace the whole collection (atomic in InMemoryStorage's lock;
        # eventual-consistent in Firestore but acceptable for an audit log).
        self.storage.clear("feedback_corrections")
        for e in entries:
            self.storage.append("feedback_corrections", e)
        return True

    def get_feedback_metrics(self) -> dict[str, Any]:
        entries = self.storage.query("feedback_corrections")
        if not entries:
            return {
                "total_corrections": 0,
                "validated_corrections": 0,
                "validation_rate": 0.0,
                "by_role": {},
            }
        validated = sum(1 for e in entries if e.get("validated"))
        by_role: dict[str, int] = {}
        for e in entries:
            r = e.get("user_role", "unknown")
            by_role[r] = by_role.get(r, 0) + 1
        return {
            "total_corrections": len(entries),
            "validated_corrections": validated,
            "validation_rate": validated / len(entries),
            "by_role": by_role,
        }
