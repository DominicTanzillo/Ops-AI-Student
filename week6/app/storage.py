"""Pluggable storage backend shared by Weeks 5-7.

Two implementations:
  - InMemoryStorage : Python dicts/lists. Default. Zero deps, zero cost,
    resets on process restart. Used for all dev + testing.
  - FirestoreStorage : google.cloud.firestore. Production-shape.
    Code is complete + importable but only instantiated when
    STORAGE_BACKEND=firestore. NOT touched in tests.

All cost tracking, audit log, rate-limit history, query cache, and
feedback corrections route through the same Storage interface so the
backend is a one-line config change.

Collection naming convention (used consistently across Weeks 5-7):
  - query_history          - Agent.query() records
  - audit_log              - AccessController access attempts
  - rate_limit_timestamps  - RateLimiter per-user query times
  - user_spending          - CostEnforcer per-user cost rollups
  - query_cache            - OptimizationStrategy cached responses
  - feedback_corrections   - FeedbackLoop user corrections

Backend selection: set STORAGE_BACKEND=firestore in .env to use Firestore.
Anything else (default 'memory') uses InMemoryStorage.
"""
from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


class Storage(ABC):
    """Abstract key/value + list storage with simple query semantics.

    Conceptually a collection-of-documents store. Two access patterns:
      - keyed: set/get/delete by (collection, key)
      - sequence: append to a collection, query the collection with a filter
    Both patterns map cleanly onto Firestore documents/subcollections AND
    onto plain Python dicts.
    """

    @abstractmethod
    def get(self, collection: str, key: str) -> Optional[Any]:
        """Return value at (collection, key) or None."""

    @abstractmethod
    def set(self, collection: str, key: str, value: Any) -> None:
        """Write value at (collection, key). Overwrites any existing value."""

    @abstractmethod
    def delete(self, collection: str, key: str) -> None:
        """Remove (collection, key); no-op if absent."""

    @abstractmethod
    def append(self, collection: str, value: Any) -> None:
        """Append value to collection as a sequence entry."""

    @abstractmethod
    def query(
        self,
        collection: str,
        filter_fn: Optional[Callable[[Any], bool]] = None,
    ) -> list:
        """Return entries in collection matching filter_fn (or all if None).

        Returns appended sequence entries; does NOT return keyed entries.
        Callers using set/get should use those, not query.
        """

    @abstractmethod
    def clear(self, collection: str) -> None:
        """Remove all entries (both keyed and sequence) from a collection."""


# ----------------------------------------------------------------------
# InMemoryStorage - default, zero-cost
# ----------------------------------------------------------------------
class InMemoryStorage(Storage):
    """Python-dict-backed storage. Thread-safe via a single lock.

    Holds two structures per collection:
      - _kv[collection][key] -> value         (for set/get/delete)
      - _seq[collection] -> [value, value, ...]  (for append/query)
    These are independent so the same collection can hold both styles
    (e.g. user_spending uses kv per-user; rate_limit_timestamps uses
    seq style or kv-of-list - we keep them separate for clarity).
    """

    def __init__(self) -> None:
        self._kv: dict[str, dict[str, Any]] = {}
        self._seq: dict[str, list[Any]] = {}
        self._lock = threading.RLock()

    def get(self, collection: str, key: str) -> Optional[Any]:
        with self._lock:
            return self._kv.get(collection, {}).get(key)

    def set(self, collection: str, key: str, value: Any) -> None:
        with self._lock:
            self._kv.setdefault(collection, {})[key] = value

    def delete(self, collection: str, key: str) -> None:
        with self._lock:
            self._kv.get(collection, {}).pop(key, None)

    def append(self, collection: str, value: Any) -> None:
        with self._lock:
            self._seq.setdefault(collection, []).append(value)

    def query(
        self,
        collection: str,
        filter_fn: Optional[Callable[[Any], bool]] = None,
    ) -> list:
        with self._lock:
            entries = list(self._seq.get(collection, []))
        if filter_fn is None:
            return entries
        return [e for e in entries if filter_fn(e)]

    def clear(self, collection: str) -> None:
        with self._lock:
            self._kv.pop(collection, None)
            self._seq.pop(collection, None)


# ----------------------------------------------------------------------
# FirestoreStorage - production-shape, lazy-imported so plain `pip install`
# without google-cloud-firestore still imports this module cleanly.
# ----------------------------------------------------------------------
class FirestoreStorage(Storage):
    """google.cloud.firestore-backed storage.

    Not instantiated unless STORAGE_BACKEND=firestore. The import of the
    Firestore client is deferred to __init__ so importing this module
    without the firestore package installed does not fail.
    """

    def __init__(self, project_id: Optional[str] = None) -> None:
        try:
            from google.cloud import firestore  # noqa: WPS433 (lazy import is intentional)
        except ImportError as e:
            raise RuntimeError(
                "FirestoreStorage requested but google-cloud-firestore "
                "is not installed. `pip install google-cloud-firestore` "
                "or set STORAGE_BACKEND=memory."
            ) from e
        self._fs = firestore.Client(project=project_id) if project_id else firestore.Client()
        log.info("FirestoreStorage initialized (project=%s)", project_id or "default")

    def get(self, collection: str, key: str) -> Optional[Any]:
        doc = self._fs.collection(collection).document(key).get()
        return doc.to_dict() if doc.exists else None

    def set(self, collection: str, key: str, value: Any) -> None:
        # Firestore documents must be dicts. Wrap scalars / lists.
        payload = value if isinstance(value, dict) else {"value": value}
        self._fs.collection(collection).document(key).set(payload)

    def delete(self, collection: str, key: str) -> None:
        self._fs.collection(collection).document(key).delete()

    def append(self, collection: str, value: Any) -> None:
        # Append by auto-id document. Wrap scalars.
        payload = value if isinstance(value, dict) else {"value": value}
        self._fs.collection(collection).add(payload)

    def query(
        self,
        collection: str,
        filter_fn: Optional[Callable[[Any], bool]] = None,
    ) -> list:
        # Pull all docs; filter client-side. The assignment volumes do not
        # justify server-side filters or pagination. Production would push
        # filter predicates to firestore.Query.where(...).
        docs = self._fs.collection(collection).stream()
        entries = [d.to_dict() for d in docs]
        if filter_fn is None:
            return entries
        return [e for e in entries if filter_fn(e)]

    def clear(self, collection: str) -> None:
        # Firestore has no native "delete collection" - iterate and delete.
        for d in self._fs.collection(collection).stream():
            d.reference.delete()


# ----------------------------------------------------------------------
# Factory: pick a backend from env
# ----------------------------------------------------------------------
def get_storage() -> Storage:
    """Return a Storage instance based on STORAGE_BACKEND env var.

    'firestore' => FirestoreStorage (requires google-cloud-firestore +
    auth). Anything else => InMemoryStorage (default; recommended for
    dev + tests + the assignment submission).
    """
    backend = os.getenv("STORAGE_BACKEND", "memory").lower()
    if backend == "firestore":
        return FirestoreStorage(project_id=os.getenv("GCP_PROJECT_ID"))
    return InMemoryStorage()
