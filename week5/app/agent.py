"""Week 5 Agent: Gemini-powered multi-shot tool-calling.

Architecture (locked in Stage 1 design):
  - Native Gemini function-calling (FunctionDeclaration schema per tool)
  - Multi-shot reasoning loop: LLM -> tools -> LLM -> ... -> final answer
  - Iteration cap (AGENT_MAX_ITERATIONS, default 5) prevents runaway loops
  - LLMClient is a Protocol so tests inject a MockLLMClient (zero cost)

The real GeminiClient wraps google.genai. It is constructed lazily so
importing this module without google.genai installed does NOT fail; only
agents that actually call Gemini need the package.

Persistence: query history goes through the storage layer (default
InMemoryStorage; Firestore-ready). get_metrics() rolls up cost across
all recorded queries.

Cost tracking: Gemini 2.5 Pro / 1.5 Flash pricing per 1M tokens, applied
from response.usage_metadata. RECOMMENDED for Week 7's CostAnalyzer to
consume the same numbers.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, runtime_checkable

from .storage import Storage, get_storage
from .tools import Tool, default_tools

log = logging.getLogger(__name__)

# Gemini pricing per 1M tokens (per Week 5 README; Flash is cheaper for the
# Week 7 model-router story). If pricing changes, update here only.
PRICING_PER_1M = {
    "gemini-2.5-pro":   {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash": {"input": 0.0375, "output": 0.15},
}
DEFAULT_MODEL = "gemini-2.5-pro"


# ----------------------------------------------------------------------
# LLM response shape (provider-agnostic)
# ----------------------------------------------------------------------
@dataclass
class FunctionCall:
    """A single tool call the LLM wants the agent to execute."""
    name: str
    args: dict[str, Any]


@dataclass
class LLMResponse:
    """Normalized response from the LLM, regardless of provider."""
    text: Optional[str] = None
    function_calls: list[FunctionCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = None

    @property
    def has_function_calls(self) -> bool:
        return bool(self.function_calls)


@runtime_checkable
class LLMClient(Protocol):
    """Anything that can run an LLM turn given a message history + tools.

    Implementations:
      - GeminiClient   : wraps google.genai
      - MockLLMClient  : in tests/conftest.py, returns canned responses
    """

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool],
        model: str,
    ) -> LLMResponse:
        ...


# ----------------------------------------------------------------------
# Real Gemini client (wraps google.genai). Lazy-imports the SDK.
# ----------------------------------------------------------------------
class GeminiClient:
    """Wraps google.genai for the agent.

    Translates our internal message format (a list of dicts) to google.genai
    Content objects, advertises Tool function-declarations, and unpacks the
    response into an LLMResponse.

    Internal message dict shape (we keep our own format so the agent can be
    persisted/replayed/tested without provider-specific types):
      {"role": "user"|"model"|"function", "text": str}             - text turn
      {"role": "model", "function_calls": [{"name": str, "args": {...}}]}
      {"role": "function", "name": str, "response": str_or_dict}
    """

    def __init__(self, api_key: str):
        try:
            from google import genai  # noqa: WPS433
        except ImportError as e:
            raise RuntimeError(
                "google-genai is required for live Gemini calls. "
                "`pip install google-genai` or inject a mock LLMClient."
            ) from e
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is empty. Set it in .env or pass api_key=.",
            )
        self._genai = genai
        self._client = genai.Client(api_key=api_key)

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool],
        model: str,
    ) -> LLMResponse:
        from google.genai import types  # noqa: WPS433

        contents = [self._to_content(m, types) for m in messages]
        gemini_tools = [
            types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters=t.parameters_schema,
                )
                for t in tools
            ]),
        ] if tools else None
        config = types.GenerateContentConfig(tools=gemini_tools) if gemini_tools else None

        resp = self._client.models.generate_content(
            model=model, contents=contents, config=config,
        )
        return self._from_response(resp)

    @staticmethod
    def _to_content(m: dict[str, Any], types) -> Any:
        role = m.get("role", "user")
        if "function_calls" in m and m["function_calls"]:
            return types.Content(
                role="model",
                parts=[
                    types.Part.from_function_call(name=fc["name"], args=fc["args"])
                    for fc in m["function_calls"]
                ],
            )
        if role == "function":
            payload = m.get("response", "")
            if isinstance(payload, str):
                # google.genai expects a dict; wrap stringified tool output
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    payload = {"result": payload}
            return types.Content(
                role="user",  # function responses are role=user in genai
                parts=[types.Part.from_function_response(
                    name=m.get("name", ""), response=payload,
                )],
            )
        return types.Content(role=role, parts=[types.Part(text=m.get("text", ""))])

    @staticmethod
    def _from_response(resp: Any) -> LLMResponse:
        out = LLMResponse(raw=resp)
        usage = getattr(resp, "usage_metadata", None)
        if usage is not None:
            out.input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            out.output_tokens = getattr(usage, "candidates_token_count", 0) or 0
        cand = (resp.candidates or [None])[0]
        if cand is None or cand.content is None:
            return out
        text_parts: list[str] = []
        for part in cand.content.parts or []:
            fc = getattr(part, "function_call", None)
            if fc is not None and getattr(fc, "name", None):
                # google.genai exposes args either as dict or MapComposite.
                args = dict(fc.args) if fc.args else {}
                out.function_calls.append(FunctionCall(name=fc.name, args=args))
            elif getattr(part, "text", None):
                text_parts.append(part.text)
        if text_parts:
            out.text = "".join(text_parts)
        return out


# ----------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------
class Agent:
    """Multi-shot LLM-tool-use agent.

    Lifecycle of a query:
      1. Build the system + user message
      2. Loop up to max_iterations:
         - call LLM with messages + tool declarations
         - if response.has_function_calls: execute each tool, append result, loop
         - else: response.text is the final answer; return
      3. If iteration cap exceeded, return a 'max_iterations_reached' warning
    """

    SYSTEM_PROMPT = (
        "You are TechCorp's internal assistant. Answer questions about "
        "employees, policies, expenses, and benefits. You have access to "
        "tools (employee_lookup, policy_search, expense_query). Call them "
        "as needed; cite the data they return in your answer. Be concise."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        tools: Optional[list[Tool]] = None,
        storage: Optional[Storage] = None,
        llm_client: Optional[LLMClient] = None,
        model: str = DEFAULT_MODEL,
        max_iterations: Optional[int] = None,
        db_path: Optional[str] = None,
    ):
        self.tools_by_name: dict[str, Tool] = {
            t.name: t for t in (tools if tools is not None else default_tools(db_path=db_path))
        }
        self.storage: Storage = storage or get_storage()
        self.model = model
        self.max_iterations = max_iterations or int(os.getenv("AGENT_MAX_ITERATIONS", "5"))
        if llm_client is not None:
            self.llm: LLMClient = llm_client
        else:
            self.llm = GeminiClient(api_key=api_key or os.getenv("GOOGLE_API_KEY", ""))

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------
    def query(
        self,
        user_query: str,
        user_id: str = "anon",
        user_role: str = "engineer",
    ) -> dict[str, Any]:
        """Answer one user question. Returns a structured result dict."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "text": f"{self.SYSTEM_PROMPT}\n\nQuestion: {user_query}"},
        ]
        tool_calls_made: list[dict[str, Any]] = []
        total_in = 0
        total_out = 0
        tool_list = list(self.tools_by_name.values())

        for i in range(1, self.max_iterations + 1):
            try:
                response = self.llm.generate(messages, tool_list, self.model)
            except Exception as e:
                log.exception("LLM call failed on iteration %d", i)
                result = self._build_result(
                    answer=f"(agent error: {e})",
                    iterations=i,
                    in_tokens=total_in,
                    out_tokens=total_out,
                    tool_calls=tool_calls_made,
                    error=str(e),
                )
                self._record_query(user_id, user_role, user_query, result)
                return result

            total_in += response.input_tokens
            total_out += response.output_tokens

            if not response.has_function_calls:
                result = self._build_result(
                    answer=response.text or "(empty response)",
                    iterations=i,
                    in_tokens=total_in,
                    out_tokens=total_out,
                    tool_calls=tool_calls_made,
                )
                self._record_query(user_id, user_role, user_query, result)
                return result

            # Record the model's tool-call turn so the next LLM call has
            # the full conversation history.
            messages.append({
                "role": "model",
                "function_calls": [
                    {"name": fc.name, "args": fc.args} for fc in response.function_calls
                ],
            })
            # Execute each tool, append response
            for fc in response.function_calls:
                tool = self.tools_by_name.get(fc.name)
                if tool is None:
                    result_str = json.dumps({"error": f"unknown tool: {fc.name}"})
                else:
                    try:
                        result_str = tool.execute(**fc.args)
                    except TypeError as e:
                        result_str = json.dumps({
                            "error": f"bad args for {fc.name}: {e}",
                            "args_received": fc.args,
                        })
                tool_calls_made.append({
                    "tool": fc.name,
                    "args": fc.args,
                    "result_length": len(result_str),
                })
                messages.append({
                    "role": "function",
                    "name": fc.name,
                    "response": result_str,
                })

        # Iteration cap exceeded
        result = self._build_result(
            answer="(agent exceeded max iterations without producing a final answer)",
            iterations=self.max_iterations,
            in_tokens=total_in,
            out_tokens=total_out,
            tool_calls=tool_calls_made,
            warning="max_iterations_reached",
        )
        self._record_query(user_id, user_role, user_query, result)
        return result

    # ------------------------------------------------------------------
    # metrics
    # ------------------------------------------------------------------
    def get_metrics(self) -> dict[str, Any]:
        """Roll up cost across all recorded queries."""
        history = self.storage.query("query_history")
        if not history:
            return {
                "total_queries": 0,
                "total_cost": 0.0,
                "avg_cost_per_query": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
            }
        total_cost = sum(h.get("cost", 0.0) for h in history)
        return {
            "total_queries": len(history),
            "total_cost": total_cost,
            "avg_cost_per_query": total_cost / len(history),
            "total_input_tokens": sum(h.get("input_tokens", 0) for h in history),
            "total_output_tokens": sum(h.get("output_tokens", 0) for h in history),
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_result(
        self,
        answer: str,
        iterations: int,
        in_tokens: int,
        out_tokens: int,
        tool_calls: list[dict[str, Any]],
        warning: Optional[str] = None,
        error: Optional[str] = None,
    ) -> dict[str, Any]:
        cost = self._estimate_query_cost(in_tokens, out_tokens, self.model)
        out: dict[str, Any] = {
            "answer": answer,
            "iterations": iterations,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "tokens_used": in_tokens + out_tokens,
            "cost": cost,
            "tool_calls": tool_calls,
            "model": self.model,
        }
        if warning:
            out["warning"] = warning
        if error:
            out["error"] = error
        return out

    @staticmethod
    def _estimate_query_cost(input_tokens: int, output_tokens: int, model: str) -> float:
        p = PRICING_PER_1M.get(model, PRICING_PER_1M[DEFAULT_MODEL])
        return (
            (input_tokens / 1_000_000) * p["input"]
            + (output_tokens / 1_000_000) * p["output"]
        )

    def _record_query(
        self,
        user_id: str,
        user_role: str,
        query: str,
        result: dict[str, Any],
    ) -> None:
        self.storage.append("query_history", {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "user_role": user_role,
            "query": query,
            "answer": (result.get("answer") or "")[:500],
            "iterations": result.get("iterations", 0),
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "cost": result.get("cost", 0.0),
            "tool_calls": result.get("tool_calls", []),
            "model": result.get("model"),
            "warning": result.get("warning"),
            "error": result.get("error"),
        })
