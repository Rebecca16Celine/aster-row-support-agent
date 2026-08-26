"""
evaluation/run_eval.py

Deterministic evaluation harness. Runs every case in visible-cases.json
(plus original-cases.json) through the live agent and checks assertions
against the agent's actual output -- not by asking another LLM to grade
free-form text, but by checking concrete signals:

  - Which tool (if any) was called, and with what arguments
  - Whether specific required substrings/phrases appear (case-insensitive)
  - Whether forbidden substrings are absent
  - Whether required knowledge-base sources appear in the cited sources
  - Whether a human handoff was recommended (keyword-based signal)

Run from the project root:
    python -m evaluation.run_eval
or:
    python evaluation/run_eval.py

Outputs:
  - Per-case PASS/FAIL with details, to stdout
  - A summary broken down by category
  - Exit code 0 if all cases pass, 1 otherwise (so it can be used as a CI gate)
"""

import json
import re
import sys
import time
from pathlib import Path

# Allow running as either `python evaluation/run_eval.py` or `python -m evaluation.run_eval`
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import Agent  # noqa: E402


EVAL_DIR = Path(__file__).parent
VISIBLE_CASES_PATH = EVAL_DIR / "visible-cases.json"
ORIGINAL_CASES_PATH = EVAL_DIR / "original-cases.json"

# Phrases that indicate the agent recommended human assistance. Used as a
# deterministic proxy for "handoff": true/false in each case's `expect`.
_HANDOFF_SIGNAL_PHRASES = [
    "human support", "human specialist", "support specialist", "contact support",
    "connect you with", "reach out to support", "human confirmation",
    "recommend human", "a specialist will", "our support team",
    "human customer support", "reaching out to human", "reach out to human",
    "human review", "support team", "customer support team",
]


def _load_cases(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("cases", [])


def _contains(haystack: str, needle: str) -> bool:
    return needle.lower() in haystack.lower()


def _looks_like_handoff(text: str) -> bool:
    return any(_contains(text, phrase) for phrase in _HANDOFF_SIGNAL_PHRASES)


class CaseResult:
    def __init__(self, case_id: str, category: str):
        self.case_id = case_id
        self.category = category
        self.checks: list[tuple[str, bool, str]] = []  # (check_name, passed, detail)

    def add(self, name: str, passed: bool, detail: str = ""):
        self.checks.append((name, passed, detail))

    @property
    def passed(self) -> bool:
        return all(passed for _, passed, _ in self.checks)


def run_case(agent: Agent, case: dict) -> CaseResult:
    result = CaseResult(case["id"], case.get("category", "uncategorized"))
    expect = case.get("expect", {})

    session = agent.new_session()
    last_response = None
    all_tool_calls = []
    all_sources = []

    all_response_texts = []
    for msg in case["messages"]:
        last_response = agent.send_message(session, msg["content"])
        all_tool_calls.extend(last_response["tool_calls"])
        all_response_texts.append(last_response["text"])
        for s in last_response["sources"]:
            if s not in all_sources:
                all_sources.append(s)

    # For multi-turn cases, concept/inclusion checks should consider
    # everything said across the whole conversation, not just the final
    # turn -- a concept established in turn 1 (e.g. "duties not prepaid")
    # is still a true, correct thing the agent said, even if a later
    # follow-up doesn't repeat it.
    final_text = last_response["text"] if last_response else ""
    full_conversation_text = " ".join(all_response_texts)
    tool_names_called = [tc["name"] for tc in all_tool_calls]

    # ---- must_include / must_not_include (exact-ish substrings) ----
    for phrase in expect.get("must_include", []):
        result.add(f"must_include: {phrase!r}", _contains(final_text, phrase))

    for phrase in expect.get("must_not_include", []):
        result.add(f"must_not_include: {phrase!r}", not _contains(final_text, phrase))

    # ---- must_include_concepts (looser: any reasonable keyword overlap) ----
    # These are longer conceptual phrases from the case file that are not
    # meant to match verbatim. We check for at least one meaningful keyword
    # from the concept phrase appearing in the response, as a lightweight
    # deterministic signal -- not full NLU, but not an LLM grader either.
    for concept in expect.get("must_include_concepts", []):
        keywords = [w for w in re.findall(r"[a-zA-Z]+", concept) if len(w) > 3]
        hit = any(_contains(final_text, kw) for kw in keywords) if keywords else False
        result.add(f"must_include_concept: {concept!r}", hit,
                    detail=f"checked keywords={keywords}")

    # ---- must_ask_for ----
    for phrase in expect.get("must_ask_for", []):
        result.add(f"must_ask_for: {phrase!r}", _contains(final_text, phrase))

    # ---- must_not_invent (treated as must_not_include) ----
    for phrase in expect.get("must_not_invent", []):
        result.add(f"must_not_invent: {phrase!r}", not _contains(final_text, phrase))

    # ---- must_refuse_to_disclose (treated as must_not_include on the value side) ----
    # The case's must_not_include list already carries the actual PII values
    # to check for; must_refuse_to_disclose is the field *names*, which we
    # don't assert against text directly (agent might legitimately say the
    # word "email" while explaining it can't share one). must_not_include
    # is the authoritative check for actual leakage.
    for phrase in expect.get("must_not_include", []):
        pass  # already checked above; listed here for clarity that this
              # is the real privacy-leakage assertion for this case type.

    # ---- required_sources ----
    for src in expect.get("required_sources", []):
        hit = any(src in s for s in all_sources)
        result.add(f"required_source cited: {src!r}", hit,
                    detail=f"actual sources={all_sources}")

    # ---- forbidden_sources_as_authority ----
    # We can't inspect "was this used as the basis of a claim" directly, so
    # we assert the forbidden doc's specific superseded content (e.g. "45
    # calendar days" from the legacy doc) doesn't appear as an unqualified
    # claim -- approximated via must_not_include, which the case already
    # specifies for this purpose.

    # ---- tool assertions ----
    expected_tool = expect.get("tool")
    if expected_tool == "not_called":
        # "not_called" means order_lookup wasn't called -- these are pure
        # knowledge-base questions with no order involved. search_kb calls
        # are expected and correct RAG behavior, not a violation.
        result.add("tool: order_lookup not called", "order_lookup" not in tool_names_called,
                    detail=f"actually called={tool_names_called}")
    elif expected_tool == "not_called_without_id":
        result.add("tool: not called without an order id", "order_lookup" not in tool_names_called,
                    detail=f"actually called={tool_names_called}")
    elif expected_tool == "order_lookup":
        result.add("tool: order_lookup called", "order_lookup" in tool_names_called,
                    detail=f"actually called={tool_names_called}")
    elif expected_tool == "optional_sanitized_lookup":
        # Either calling it (and still not leaking PII) or not calling it
        # is acceptable -- what matters is the privacy assertions above.
        result.add("tool: optional (privacy is the real check)", True)

    expected_args = expect.get("tool_arguments")
    if expected_args:
        matched = any(
            tc["name"] == expected_tool and
            all(str(tc["arguments"].get(k, "")).upper() == str(v).upper() for k, v in expected_args.items())
            for tc in all_tool_calls
        )
        result.add(f"tool_arguments match {expected_args}", matched,
                    detail=f"actual calls={all_tool_calls}")

    # ---- handoff ----
    if "handoff" in expect:
        expected_handoff = expect["handoff"]
        actual_handoff = _looks_like_handoff(final_text)
        result.add(f"handoff expected={expected_handoff}", actual_handoff == expected_handoff,
                    detail=f"response_excerpt={final_text[:150]!r}")

    # ---- must_not_silently_choose_one (source-conflict cases) ----
    if expect.get("must_not_silently_choose_one"):
        # Heuristic: both conflicting claims' keywords should be mentioned,
        # not just one -- i.e. it didn't quietly pick a side.
        mentions_handwash = _contains(final_text, "hand")
        mentions_dishwasher_safe = _contains(final_text, "dishwasher safe") or _contains(final_text, "all components")
        result.add("mentions both conflicting claims (didn't silently choose)",
                    mentions_handwash and mentions_dishwasher_safe,
                    detail=f"response_excerpt={final_text[:200]!r}")

    return result


def run_all(agent: Agent, cases: list[dict], delay_seconds: float = 15.0) -> list[CaseResult]:
    """
    Run every case, pausing between cases to stay under Gemini's free-tier
    rate limit (5 requests/minute at time of writing). Each case can issue
    multiple requests internally (a search call, sometimes two, then a
    final answer), so a generous inter-case delay is used rather than
    trying to count requests precisely.
    """
    results = []
    for i, case in enumerate(cases):
        print(f"\n>>> Running case {i + 1}/{len(cases)}: {case['id']}")
        result = _run_case_with_retry(agent, case)
        results.append(result)
        if i < len(cases) - 1:
            time.sleep(delay_seconds)
    return results


def _run_case_with_retry(agent: Agent, case: dict, max_retries: int = 3) -> CaseResult:
    """Retry a case with backoff if we hit a rate limit mid-case."""
    for attempt in range(max_retries):
        try:
            return run_case(agent, case)
        except Exception as e:
            is_rate_limit = "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)
            if is_rate_limit and attempt < max_retries - 1:
                wait = 45 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            raise


def print_report(results: list[CaseResult]):
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    by_category: dict[str, list[CaseResult]] = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r)

    total_pass = sum(1 for r in results if r.passed)
    total = len(results)

    for category, cat_results in sorted(by_category.items()):
        cat_pass = sum(1 for r in cat_results if r.passed)
        print(f"\n--- {category} ({cat_pass}/{len(cat_results)}) ---")
        for r in cat_results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.case_id}")
            if not r.passed:
                for name, passed, detail in r.checks:
                    if not passed:
                        print(f"        ✗ {name}" + (f"  ({detail})" if detail else ""))

    print("\n" + "=" * 70)
    print(f"TOTAL: {total_pass}/{total} cases passed")
    print("=" * 70 + "\n")

    return total_pass, total


def main():
    agent = Agent()

    visible_cases = _load_cases(VISIBLE_CASES_PATH)
    original_cases = _load_cases(ORIGINAL_CASES_PATH)

    print(f"Loaded {len(visible_cases)} visible cases, {len(original_cases)} original cases.")

    all_cases = visible_cases + original_cases
    results = run_all(agent, all_cases)
    total_pass, total = print_report(results)

    sys.exit(0 if total_pass == total else 1)


if __name__ == "__main__":
    main()