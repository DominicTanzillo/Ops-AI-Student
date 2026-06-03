"""Unit tests for the storage abstraction layer.

InMemoryStorage is fully exercised; FirestoreStorage is import-tested only
(no real Firestore calls in this suite - that would require credentials
+ network + cost).
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from app.storage import (
    InMemoryStorage,
    Storage,
    get_storage,
)


# ----------------------------------------------------------------------
# InMemoryStorage - basic key/value
# ----------------------------------------------------------------------
def test_set_get_roundtrip():
    s = InMemoryStorage()
    s.set("user_spending", "alice", 12.34)
    assert s.get("user_spending", "alice") == 12.34


def test_get_missing_returns_none():
    s = InMemoryStorage()
    assert s.get("user_spending", "nobody") is None


def test_set_overwrites():
    s = InMemoryStorage()
    s.set("k", "a", 1)
    s.set("k", "a", 2)
    assert s.get("k", "a") == 2


def test_delete_removes_key():
    s = InMemoryStorage()
    s.set("k", "a", 1)
    s.delete("k", "a")
    assert s.get("k", "a") is None


def test_delete_missing_is_noop():
    s = InMemoryStorage()
    s.delete("k", "absent")  # should not raise


# ----------------------------------------------------------------------
# InMemoryStorage - sequence append/query
# ----------------------------------------------------------------------
def test_append_and_query_all():
    s = InMemoryStorage()
    s.append("audit_log", {"role": "engineer", "allowed": True})
    s.append("audit_log", {"role": "engineer", "allowed": False})
    s.append("audit_log", {"role": "hr", "allowed": True})
    assert len(s.query("audit_log")) == 3


def test_query_with_filter():
    s = InMemoryStorage()
    s.append("audit_log", {"role": "engineer", "allowed": True})
    s.append("audit_log", {"role": "hr", "allowed": True})
    s.append("audit_log", {"role": "engineer", "allowed": False})
    eng = s.query("audit_log", lambda e: e["role"] == "engineer")
    assert len(eng) == 2


def test_query_empty_collection_returns_empty():
    s = InMemoryStorage()
    assert s.query("never_written") == []


def test_kv_and_seq_are_independent():
    """A collection can hold BOTH a kv side and a seq side without interference."""
    s = InMemoryStorage()
    s.set("mixed", "kv_key", "kv_value")
    s.append("mixed", {"seq": 1})
    s.append("mixed", {"seq": 2})
    assert s.get("mixed", "kv_key") == "kv_value"
    assert len(s.query("mixed")) == 2  # query only returns seq entries


def test_clear_removes_both():
    s = InMemoryStorage()
    s.set("c", "k", "v")
    s.append("c", "item")
    s.clear("c")
    assert s.get("c", "k") is None
    assert s.query("c") == []


# ----------------------------------------------------------------------
# Thread safety smoke test - concurrent appends do not lose entries
# ----------------------------------------------------------------------
def test_concurrent_appends_no_loss():
    s = InMemoryStorage()
    N = 100

    def writer(start: int) -> None:
        for i in range(N):
            s.append("counts", start + i)

    threads = [threading.Thread(target=writer, args=(t * N,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(s.query("counts")) == 8 * N


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------
def test_get_storage_defaults_to_memory(monkeypatch):
    monkeypatch.delenv("STORAGE_BACKEND", raising=False)
    s = get_storage()
    assert isinstance(s, InMemoryStorage)


def test_get_storage_unknown_backend_defaults_to_memory(monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "nonsense")
    s = get_storage()
    assert isinstance(s, InMemoryStorage)


def test_get_storage_firestore_without_lib_raises(monkeypatch):
    """If user sets STORAGE_BACKEND=firestore but the lib is missing, fail loud
    rather than silently falling back to memory."""
    monkeypatch.setenv("STORAGE_BACKEND", "firestore")
    # Force the import path inside FirestoreStorage.__init__ to fail by
    # blocking the module name from being importable. We use a stub that
    # mimics ImportError on the deferred import.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *a, **kw):
        if name == "google.cloud" or name.startswith("google.cloud.firestore"):
            raise ImportError(f"blocked for test: {name}")
        return real_import(name, *a, **kw)

    if isinstance(__builtins__, dict):
        monkeypatch.setitem(__builtins__, "__import__", fake_import)
    else:
        monkeypatch.setattr(__builtins__, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="FirestoreStorage requested"):
        get_storage()


# ----------------------------------------------------------------------
# Storage is an ABC - cannot instantiate directly
# ----------------------------------------------------------------------
def test_storage_is_abstract():
    with pytest.raises(TypeError):
        Storage()  # type: ignore[abstract]
