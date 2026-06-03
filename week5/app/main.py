"""FastAPI surface for the Week 5 agent.

Endpoints:
  POST /agent/query    - run a user question through the agent
  GET  /agent/metrics  - cost + token rollup across recorded queries
  GET  /health         - liveness probe

The Agent instance is constructed once at startup. For tests, the
agent is monkey-patched via a module-level singleton so a MockLLMClient
can be injected without a real Gemini key.

Local dev:
  uvicorn app.main:app --reload
  open http://localhost:8000/docs

.env loading: app/main.py reads GOOGLE_API_KEY (and STORAGE_BACKEND, etc.)
from a .env file in the working directory via python-dotenv. The .env
itself is gitignored.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import Agent

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(
    title="TechCorp Agent (Week 5)",
    description="LLM-powered Q&A agent over TechCorp data.",
    version="0.1.0",
)


# ----------------------------------------------------------------------
# Agent singleton - lazy-constructed so importing this module does NOT
# instantiate a GeminiClient (which needs a real API key). Tests can
# inject their own Agent via set_agent().
# ----------------------------------------------------------------------
_agent: Optional[Agent] = None


def get_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent()  # uses GOOGLE_API_KEY + default tools
    return _agent


def set_agent(agent: Agent) -> None:
    """Inject a pre-built Agent (used by tests with MockLLMClient)."""
    global _agent
    _agent = agent


# ----------------------------------------------------------------------
# Request / response schemas
# ----------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., description="User's natural-language question")
    user_id: str = Field("anon", description="Stable user identifier (for cost + rate-limit tracking)")
    user_role: str = Field("engineer", description="Role: engineer|manager|hr|finance|executive")


class ToolCallTrace(BaseModel):
    tool: str
    args: dict[str, Any]
    result_length: int


class QueryResponse(BaseModel):
    answer: str
    iterations: int
    input_tokens: int
    output_tokens: int
    tokens_used: int
    cost: float
    tool_calls: list[ToolCallTrace]
    model: str
    warning: Optional[str] = None
    error: Optional[str] = None


class MetricsResponse(BaseModel):
    total_queries: int
    total_cost: float
    avg_cost_per_query: float
    total_input_tokens: int
    total_output_tokens: int


class HealthResponse(BaseModel):
    status: str
    has_api_key: bool
    storage_backend: str
    model: str


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@app.post("/agent/query", response_model=QueryResponse)
def query_agent(req: QueryRequest) -> QueryResponse:
    try:
        agent = get_agent()
    except (ValueError, RuntimeError) as e:
        # No API key or no google-genai installed
        raise HTTPException(status_code=503, detail=str(e))
    result = agent.query(req.question, user_id=req.user_id, user_role=req.user_role)
    return QueryResponse(**result)


@app.get("/agent/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    try:
        agent = get_agent()
    except (ValueError, RuntimeError):
        # Agent not initialized - return zeros so dashboards keep working
        return MetricsResponse(
            total_queries=0, total_cost=0.0, avg_cost_per_query=0.0,
            total_input_tokens=0, total_output_tokens=0,
        )
    return MetricsResponse(**agent.get_metrics())


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        has_api_key=bool(os.getenv("GOOGLE_API_KEY")),
        storage_backend=os.getenv("STORAGE_BACKEND", "memory"),
        model=os.getenv("AGENT_MODEL", "gemini-2.5-pro"),
    )
