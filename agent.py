"""
agent.py

The main agent loop: wires together
  - Gemini (via google-genai SDK) as the LLM
  - search_kb() and order_lookup() as callable tools
  - per-session conversation memory
  - structured logging for observability

Run this directly for an interactive CLI chat:
    python agent.py

Environment:
    GEMINI_API_KEY must be set (see .env.example).
"""

import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field

from google import genai
from google.genai import types
from dotenv import load_dotenv

from order_lookup import order_lookup
from search_kb import search_kb
from system_prompt import SYSTEM_PROMPT

load_dotenv()  # reads .env in the project root, if present


MODEL_NAME = "gemini-3.6-flash"

# ---- Logging (observability) ----------------------------------------------
# Plain structured logs to stderr as JSON lines, so stdout stays clean for
# the user-facing chat transcript. Never logs secrets (API key is never
# logged; tool results are logged but PII/internal fields never reach this
# code in the first place -- see order_lookup.py's field allowlist).

logger = logging.getLogger("aster_row_agent")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)


def log_event(event_type: str, **fields):
    entry = {"event": event_type, "ts": time.time(), **fields}
    logger.info(json.dumps(entry, default=str))


# ---- Tool declarations (Gemini function-calling format) -------------------

SEARCH_KB_DECLARATION = {
    "name": "search_kb",
    "description": (
        "Search the Aster & Row knowledge base (returns policy, shipping policy, "
        "warranty, product care, etc.) for passages relevant to a customer question. "
        "Returns ranked passages with source filename, heading, and authority metadata."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A short search query describing what information is needed.",
            }
        },
        "required": ["query"],
    },
}

ORDER_LOOKUP_DECLARATION = {
    "name": "order_lookup",
    "description": (
        "Look up the current status of a customer order by its order ID "
        "(format: ORD-####). Returns only customer-safe fields, or a "
        "not-found/malformed/missing signal. Never returns customer PII."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The order ID as given by the customer, e.g. 'ORD-1007'.",
            }
        },
        "required": ["order_id"],
    },
}

TOOLS = types.Tool(function_declarations=[SEARCH_KB_DECLARATION, ORDER_LOOKUP_DECLARATION])


def _execute_tool_call(name: str, args: dict, session_id: str) -> dict:
    """
    Dispatch a model-requested tool call to the real Python function and
    return a JSON-serializable result. This is the ONLY place tool calls
    are actually executed -- the model can request them, but never runs
    them directly.
    """
    if name == "search_kb":
        query = args.get("query", "")
        results = search_kb(query, top_k=5)
        log_event(
            "tool_call",
            session_id=session_id,
            tool="search_kb",
            arguments={"query": query},
            result_summary=[{"source": r["source_label"], "score": r["score"],
                              "status": r["status"]} for r in results],
        )
        return {"results": results}

    if name == "order_lookup":
        order_id = args.get("order_id", "")
        result = order_lookup(order_id)
        # sanitized result is exactly what's logged -- order_lookup() already
        # strips PII/internal fields before this point, so there is nothing
        # further to redact here.
        log_event(
            "tool_call",
            session_id=session_id,
            tool="order_lookup",
            arguments={"order_id": order_id},
            result_summary=result,
        )
        return result

    log_event("tool_call_error", session_id=session_id, tool=name, error="unknown_tool")
    return {"error": f"unknown tool: {name}"}


@dataclass
class Session:
    """Holds per-session conversation state (multi-turn memory)."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    history: list = field(default_factory=list)  # list of types.Content


class Agent:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or export GEMINI_API_KEY in your shell."
            )
        self.client = genai.Client(api_key=api_key)

    def new_session(self) -> Session:
        session = Session()
        log_event("session_start", session_id=session.session_id)
        return session

    def send_message(self, session: Session, user_message: str) -> dict:
        """
        Send one user message within `session`, run the tool-calling loop
        until the model produces a final text answer, and return a dict:
          {
            "text": final answer string,
            "tool_calls": [{"name": ..., "arguments": ..., "result": ...}, ...],
            "sources": [source_label, ...]   (deduped, from any search_kb calls made),
          }
        """
        log_event("user_message", session_id=session.session_id, text=user_message)

        session.history.append(
            types.Content(role="user", parts=[types.Part(text=user_message)])
        )

        tool_calls_made = []
        sources_seen = []

        # Loop: the model may request 0+ tool calls before giving a final answer.
        # Cap iterations defensively so a misbehaving loop can't hang forever.
        for _ in range(6):
            response = self.client.models.generate_content(
                model=MODEL_NAME,
                contents=session.history,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=[TOOLS],
                ),
            )

            candidate = response.candidates[0]
            parts = candidate.content.parts

            function_calls = [p.function_call for p in parts if p.function_call]

            if not function_calls:
                final_text = "".join(p.text for p in parts if p.text)
                session.history.append(candidate.content)
                log_event("final_response", session_id=session.session_id, text=final_text)
                return {
                    "text": final_text,
                    "tool_calls": tool_calls_made,
                    "sources": sources_seen,
                }

            # Model requested one or more tool calls: execute each, append
            # both the model's function-call turn and our function-response
            # turn to history, then loop again so the model can use results.
            session.history.append(candidate.content)

            function_response_parts = []
            for fc in function_calls:
                args = dict(fc.args) if fc.args else {}
                result = _execute_tool_call(fc.name, args, session.session_id)
                tool_calls_made.append({"name": fc.name, "arguments": args, "result": result})

                if fc.name == "search_kb":
                    for r in result.get("results", []):
                        if r["source_label"] not in sources_seen:
                            sources_seen.append(r["source_label"])

                function_response_parts.append(
                    types.Part.from_function_response(name=fc.name, response=result)
                )

            session.history.append(types.Content(role="user", parts=function_response_parts))

        # Safety valve: if we somehow looped too many times without a final
        # answer, fail closed with a handoff rather than hanging or guessing.
        log_event("loop_limit_exceeded", session_id=session.session_id)
        return {
            "text": "I'm having trouble completing this request. Let me connect you with a human support specialist.",
            "tool_calls": tool_calls_made,
            "sources": sources_seen,
        }


# ---- Interactive CLI --------------------------------------------------

def main():
    try:
        agent = Agent()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    session = agent.new_session()
    print("Aster & Row Support Agent (type 'exit' to quit)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        result = agent.send_message(session, user_input)

        print(f"\nAgent: {result['text']}")
        if result["sources"]:
            print(f"\nSources: {', '.join(result['sources'])}")
        print()


if __name__ == "__main__":
    main()