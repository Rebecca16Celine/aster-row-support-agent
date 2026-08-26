# Aster & Row Support Agent

A reliability-focused RAG customer support agent built for the AI Agent Intern take-home assignment. It answers policy questions grounded in a markdown knowledge base, looks up order status via a tool call, maintains multi-turn conversation context, and is designed to resist prompt injection, protect customer privacy, and abstain or hand off to a human rather than guess.

## Demo

*(GIF/video link goes here once recorded — see "Known limitations" for what's covered.)*

---

## Setup

### Requirements
- Python 3.10+
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com) (no billing/credit card required)

### Install

```bash
git clone <this-repo-url>
cd aster-row-support-agent
pip install google-genai python-dotenv pyyaml
```

### Configure

Copy `.env.example` to `.env` and add your key:

```
GEMINI_API_KEY=your-key-here
```

### Run the agent (interactive CLI)

```bash
python agent.py
```

Type a question, get a grounded answer with sources. Type `exit` to quit.

### Run the evaluation suite

```bash
python evaluation/run_eval.py
```

This runs 20 test cases (15 supplied + 5 original) against the live agent and prints a pass/fail report by category. **Note:** this makes real API calls and is throttled with a 15-second delay between cases to stay under Gemini's free-tier rate limits — a full run takes roughly 8–10 minutes.

---

## Model, embedding, framework, and storage choices

| Decision | Choice | Why |
|---|---|---|
| **LLM** | Google Gemini (`gemini-2.5-flash-lite` at time of writing) via the `google-genai` SDK | Free-tier API access with no cost, no card required, sufficient function-calling support for this project's needs. (Originally built against the Anthropic API; switched to Gemini for cost reasons — see "AI tools used" below.) |
| **Retrieval** | Custom TF-IDF-style keyword scoring with light stemming, title-match boosting, and authority/status weighting — no vector database | The knowledge base is 14 short files. A full embedding pipeline would add real setup cost (a vector DB, an embedding model/API, index management) with no meaningful benefit at this scale, and a transparent keyword scorer is much easier to debug and reason about for a reliability-focused assignment. |
| **Framework** | None — plain Python, direct API calls | No LangChain/LlamaIndex layer. At this scope, a framework would add abstraction without reducing real complexity, and direct control over the tool-calling loop made debugging (see bug diary) much faster. |
| **Storage** | Flat files: markdown for the knowledge base, JSON for orders, no database | Matches the assignment's data as supplied; no persistence is needed since orders.json is a read-only mock snapshot. |
| **Interface** | CLI (`python agent.py`) | Meets the "minimal interface" requirement; visual polish wasn't a scoring factor. |

---

## Architecture

```
User input
    │
    ▼
agent.py (Agent.send_message)
    │  loads system_prompt.py, maintains per-session conversation history
    ▼
Gemini API (function-calling loop, up to 6 iterations)
    │
    ├── may call search_kb(query)  ──► search_kb.py
    │       reads knowledge-base/*.md, splits into chunks by heading,
    │       keeps front-matter metadata, scores by keyword relevance
    │       (with authority/status weighting), returns ranked passages
    │
    └── may call order_lookup(order_id)  ──► order_lookup.py
            reads data/orders.json, normalizes the ID, returns ONLY
            an allowlisted set of customer-safe fields (PII and
            internal fields are stripped in Python before the result
            is ever returned — never sent to the model)
    │
    ▼
Final grounded text response + cited sources + structured JSON logs (stderr)
```

**Key design choices:**

- **Tool results are never trusted as instructions.** Both `search_kb` and `order_lookup` return plain data; `system_prompt.py` explicitly tells the model to disregard any instruction-like text found inside retrieved content or tool results, regardless of framing.
- **PII/internal fields are stripped in Python, not by asking the model nicely.** `order_lookup.py` uses an *allowlist* (not a denylist) of safe fields — new fields added to the dataset in the future are excluded by default unless explicitly added to the allowlist. Customer name, email, address, and everything under `internal` never reach the model's context window at all.
- **Retrieval surfaces conflicts instead of hiding them.** `search_kb.py` demotes (but does not delete) superseded/draft/non-official documents, and returns `status`/`policy_authority` metadata with every result so the system prompt can decide how to handle genuine conflicts between two active official sources.
- **Conversation memory** is a simple per-session list of turns (`Session.history`), passed back to the model on every call — no external memory store needed at this scale.
- **Observability**: every tool call and its (already-sanitized) result, plus every user message and final response, is logged as a structured JSON line to stderr. No secrets are ever logged.

---

## Evaluation results

Run with: `python evaluation/run_eval.py`

### Baseline (first full run)

The first complete run — before any fixes — passed **10/20** cases. On inspection, most of these "failures" were actually bugs in the eval harness's assertions, not the agent: the harness assumed policy questions would be answered with *zero* tool calls, when correctly calling `search_kb` before answering is expected RAG behavior. A few were genuine gaps.

### Final result

**17/20 cases passed.**

| Category | Passed |
|---|---|
| abstention | 1/1 |
| conversation | 1/1 |
| groundedness | 1/2 |
| multi-source-grounding | 1/1 |
| privacy | 1/1 |
| prompt-security | 1/2 |
| retrieval | 2/3 |
| source-conflict | 1/1 |
| tool-reliability | 6/6 |
| tool-use | 2/2 |

**The 3 remaining failures are documented, understood, and low-severity** — see "Known limitations" below. None involve privacy leakage, incorrect facts, or unresolved conflicts; all three are measurement-precision issues in the eval harness's keyword-based assertions, not actual agent errors (verified by manual inspection of the transcripts).

---

## Bug diary

Failures found during development, in the order discovered.

### 1. Retrieval missed "dishwasher" when the query said "dishwash"

- **Reproduction:** `search_kb("can I dishwash the breeze tumbler")` did not return `12-breeze-tumbler-product-card.md`'s "Cleaning" section (the one containing "all components are dishwasher safe") in the top results.
- **Root cause:** exact-token keyword matching — the tokenized query term `"dishwash"` never equals the tokenized document term `"dishwasher"`.
- **Fix:** added lightweight prefix-based stem matching (`_stem_match`) between query and document terms, so a shared prefix of sufficient length counts as a match.
- **Regression test:** `original-stem-match-dishwasher-wording` in `evaluation/original-cases.json` — asserts both `11-product-care.md` and `12-breeze-tumbler-product-card.md` are cited for this exact phrasing.

### 2. TrailPlus-specific policy outranked by the generic policy for a TrailPlus-specific question

- **Reproduction:** `search_kb("My TrailPlus membership was active when I ordered. What is my return window?")` ranked `01-returns-policy-current.md`'s generic "Standard return window" chunk above `09-trailplus-membership.md`'s tier-specific "Return window" chunk.
- **Root cause:** common words ("return," "window") scored similarly across both documents; nothing weighted the explicit mention of "TrailPlus" toward the TrailPlus-specific document.
- **Fix:** added a title-match bonus — a query term matching a word in a document's own front-matter `title` gets a scoring boost, so an explicit product/program name in the query strongly favors the matching document.
- **Regression test:** `trailplus-return-window` (supplied case) — asserts `09-trailplus-membership.md` is a required cited source.

### 3. Zero results for a valid, in-scope question ("Can you ship an Atlas Weekender to Germany?")

- **Reproduction:** the query above returned **zero** search results, even though the knowledge base does contain the relevant answer (Aster & Row ships internationally only to Canada).
- **Root cause:** the stem-matching minimum prefix length (originally 5 characters) excluded the word "ship" (4 characters) from matching "shipping"/"shipped" in the documents.
- **Fix:** lowered the stem-match minimum length to 4 characters, verified this didn't introduce false-positive matches on other test queries.
- **Regression test:** `unsupported-country` (supplied case) — asserts `06-international-shipping.md` is a required cited source.

### 4. Vocabulary mismatch between customer symptom-language and policy category-language

- **Reproduction:** `search_kb("final sale broken zipper warranty return")` did not surface `04-damaged-or-wrong-items.md` at all, even though that document is exactly the relevant policy — because it never uses the words "broken" or "zipper," only abstract terms like "damaged" and "defective."
- **Root cause:** customers describe *symptoms* ("broken zipper," "torn strap"); policy documents describe *categories* ("damaged," "defective"). No amount of stemming bridges genuinely different words. (This is discovered independently of the visible test cases — the supplied case for this scenario used the word "broken," but only in a compound context that happened to still retrieve correctly; testing the isolated "zipper" symptom-word surfaced the gap.)
- **Fix:** added a small, explicit synonym-expansion map applied only to the *query* side before scoring (e.g. "broken," "zipper," "torn," "cracked" → also search for "damaged"/"defective"). Source documents are never modified, per the assignment's rules.
- **Regression test:** `final-sale-damaged-exception` (supplied case) — asserts both `03-final-sale-and-promotions.md` and `04-damaged-or-wrong-items.md` are required cited sources.
- **Note:** this fix is deliberately narrow (a handful of hand-picked synonyms) rather than comprehensive. A production system would use semantic embeddings, which handle this class of gap generally instead of requiring a maintained synonym list. See "Known limitations."

### 5. Agent cited a source it hadn't actually retrieved in that turn

- **Reproduction:** on the prompt-injection test case, the agent's response correctly refused the injected instruction and stated the correct 30-day policy with a citation to `01-returns-policy-current.md` — but the trace showed **no `search_kb` call happened in that turn at all**. The fact was right; the citation was to a source not actually retrieved in that response.
- **Root cause:** the model appeared confident enough in a previously-established fact (from earlier in the same evaluation run's pattern of similar questions) to answer and cite without a fresh tool call.
- **Fix:** strengthened `system_prompt.py`'s grounding rules to explicitly require a `search_kb` call for every policy-fact question, every time, regardless of the model's apparent confidence from prior context, and explicitly frame citing an un-retrieved source as a grounding violation even when the underlying fact is correct.
- **Regression test:** covered by re-running `retrieved-prompt-injection`; the fix was verified to produce a `search_kb` call on every subsequent run of this case.

---

## Known limitations

- **Keyword-based handoff detection has both false positives and false negatives.** The eval harness detects a "handoff" by scanning the response for phrases like "human support" or "contact a specialist." This misfires when the agent mentions a human specialist as an *optional* next step for a related but different action (e.g. warranty claim approval) while still fully answering the asked question — the harness flags this as an unwanted handoff even though the core answer was complete and correct. A production-grade eval would use a structured signal (e.g. the agent explicitly setting a `handoff: true/false` field in a structured response) rather than inferring intent from prose.
- **Retrieval is keyword-based, not semantic.** The custom scorer (TF-IDF-style + light stemming + a small hand-picked synonym list) works well for this 14-document corpus but doesn't generalize the way embeddings would. Genuinely novel phrasing that doesn't share word roots or synonyms with the source documents may not retrieve correctly. At production scale, this would be replaced with an embedding-based retriever.
- **No conversation-level session expiry or persistence.** Sessions live only in memory for the lifetime of the CLI process; there's no timeout, no multi-user session isolation beyond in-memory dictionaries, and no persistence across restarts. Fine for this assignment's scope; would need addressing for a real deployment.
- **Rate-limited by Gemini's free tier.** The evaluation suite includes deliberate delays and retry logic to work within free-tier daily/per-minute quotas, which vary by model and have shifted during development (see bug diary and commit history). A paid tier or a different provider would remove this constraint.
- **Exact-substring assertions in the eval harness are sometimes stricter than necessary.** For example, one case expects the literal substring "45 calendar days" but the agent correctly said "45-calendar-day" (a grammatically valid hyphenated variant). This is a harness precision issue, not an agent correctness issue, and is left undocumented-as-fixed deliberately to avoid over-fitting the harness to one exact phrasing at the risk of missing real issues elsewhere.
- **The synonym-expansion list (bug diary #4) is manually curated and non-exhaustive.** It covers the specific gap found during testing but would need ongoing maintenance to keep up with new phrasing patterns; an embeddings-based retriever would not have this maintenance burden.

## What I'd improve before production

- Replace keyword retrieval with an embeddings-based vector search (still small-scale, e.g. a local FAISS index) to remove the vocabulary-mismatch class of bug generally rather than patching individual gaps.
- Move conversation sessions to a real store (Redis or similar) with expiry, to support multiple concurrent users safely.
- Add a proper CI-integrated eval gate (the harness already exits with code 1 on any failure, so this is close) and track eval pass rate over time.
- Replace the keyword-based handoff detector in the eval harness with a structured field the agent sets explicitly, rather than inferring it from response text.
- Add rate-limit-aware retries and a paid-tier or multi-provider fallback for production reliability.

---

## AI coding tools used

- **Claude (Anthropic, web chat interface)** was used for the majority of this build: designing the architecture, writing `order_lookup.py`, `search_kb.py`, `system_prompt.py`, `agent.py`, and `evaluation/run_eval.py`, debugging retrieval issues, and diagnosing rate-limit/model-availability errors from the Gemini API.
- **Example of an AI-generated suggestion that was wrong or incomplete:** the initial `agent.py` was written against `gemini-2.0-flash`, which turned out to be a deprecated model name no longer served by the API (Google's error message pointed to a newer model instead). The suggested replacement, `gemini-3.6-flash`, turned out to have an unexpectedly low 20-requests/day free-tier quota for new accounts — not something that was verifiable without actually hitting the limit in practice, since published documentation on Gemini's free-tier quotas was inconsistent across sources at the time. The actual fix (`gemini-2.5-flash-lite`) was found by cross-referencing multiple sources and confirming against the live error messages returned by the API itself, rather than trusting any single documentation source.

---

## Repository structure

```
.
├── README.md
├── .env.example
├── agent.py                    # main agent loop (Gemini + tools + memory + logging)
├── order_lookup.py             # order lookup tool (PII/internal-field stripping)
├── search_kb.py                # knowledge-base retrieval tool
├── system_prompt.py            # the agent's system prompt
├── knowledge-base/             # supplied, unmodified
├── data/                       # supplied, unmodified
└── evaluation/
    ├── visible-cases.json      # supplied
    ├── original-cases.json     # 5 original cases added
    └── run_eval.py              # eval harness
```