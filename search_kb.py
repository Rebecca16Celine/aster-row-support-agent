"""
search_kb.py

Retrieval over the markdown knowledge base.

Design decisions:
- No vector DB / embeddings needed at this corpus size (14 short files).
  We use TF-IDF-ish keyword scoring, which is transparent, deterministic,
  and easy to debug/test -- important for an assignment that's graded on
  reliability, not on using the fanciest retrieval tech.
- Documents are split into chunks by markdown heading (##). Each chunk
  keeps a copy of its parent document's front-matter metadata (status,
  policy_authority, effective_date, supersedes/superseded_by, etc).
- Retrieval does NOT silently drop superseded/non-authoritative chunks.
  Instead it returns them with their metadata intact, but ranks them
  lower, and exposes `status` / `policy_authority` so the calling agent
  logic (system prompt) can decide how to treat them -- e.g. explicitly
  telling the customer "this is outdated" rather than just hiding it and
  hoping nothing goes wrong.
- A chunk is never treated as an instruction. This module returns plain
  text + metadata; nothing here executes or interprets chunk content.
"""

import re
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml  # pip install pyyaml


KB_DIR = Path(__file__).parent / "knowledge-base"

# Common English stopwords -- small hand-rolled list, no external NLP dep needed.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if", "then", "else", "for", "to", "of", "in",
    "on", "at", "by", "with", "from", "as", "it", "this", "that", "these",
    "those", "i", "you", "your", "my", "we", "our", "do", "does", "did",
    "can", "could", "will", "would", "should", "may", "might", "must",
    "not", "no", "so", "how", "what", "when", "where", "why", "which",
    "have", "has", "had", "am", "about", "get", "got",
}


@dataclass
class Chunk:
    doc_id: str            # e.g. "01-returns-policy-current.md"
    heading: str            # e.g. "Standard return window"
    text: str               # the chunk's body text (heading + paragraphs)
    metadata: dict = field(default_factory=dict)  # front matter, copied

    @property
    def status(self) -> str:
        return self.metadata.get("status", "unknown")

    @property
    def policy_authority(self) -> str:
        return self.metadata.get("policy_authority", "unknown")

    def source_label(self) -> str:
        """Human-readable citation, e.g. '01-returns-policy-current.md > Standard return window'."""
        return f"{self.doc_id} > {self.heading}" if self.heading else self.doc_id


def _parse_front_matter(raw_text: str) -> tuple[dict, str]:
    """Split a markdown file into (front_matter_dict, body_text)."""
    if raw_text.startswith("---"):
        parts = raw_text.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1]
            body = parts[2]
            try:
                metadata = yaml.safe_load(fm_text) or {}
            except yaml.YAMLError:
                metadata = {}
            return metadata, body
    return {}, raw_text


def _split_into_chunks(doc_id: str, metadata: dict, body: str) -> list[Chunk]:
    """
    Split body text on '## Heading' boundaries. Text before the first
    '##' heading (e.g. the '# Title' line and any intro) becomes its own
    chunk with an empty heading.
    """
    # Find all "## Heading" lines and their positions
    heading_pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(body))

    chunks = []

    if not matches:
        text = body.strip()
        if text:
            chunks.append(Chunk(doc_id=doc_id, heading="", text=text, metadata=metadata))
        return chunks

    # Text before the first heading (title / intro line)
    intro = body[: matches[0].start()].strip()
    intro = re.sub(r"^#\s+.+$", "", intro, flags=re.MULTILINE).strip()
    if intro:
        chunks.append(Chunk(doc_id=doc_id, heading="", text=intro, metadata=metadata))

    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section_text = body[start:end].strip()
        full_text = f"{heading}\n{section_text}"
        chunks.append(Chunk(doc_id=doc_id, heading=heading, text=full_text, metadata=metadata))

    return chunks


def _load_all_chunks() -> list[Chunk]:
    chunks = []
    for path in sorted(KB_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        metadata, body = _parse_front_matter(raw)
        chunks.extend(_split_into_chunks(path.name, metadata, body))
    return chunks


_ALL_CHUNKS: list[Chunk] = _load_all_chunks()


# ---- Scoring -------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


def _build_idf(chunks: list[Chunk]) -> dict[str, float]:
    """Inverse document frequency across all chunks, for basic TF-IDF scoring."""
    n_docs = len(chunks)
    df: dict[str, int] = {}
    for c in chunks:
        seen = set(_tokenize(c.text))
        for term in seen:
            df[term] = df.get(term, 0) + 1
    return {term: math.log((n_docs + 1) / (count + 1)) + 1 for term, count in df.items()}


_IDF = _build_idf(_ALL_CHUNKS)


_STEM_MIN_LEN = 4  # only stem-match words long enough that a prefix is meaningful
                     # (4, not 5: common short-but-specific words like "ship" -> "shipping"/"shipped"
                     # need to match; below this we'd risk false positives like "for" -> "forward")


def _stem_match(query_term: str, chunk_term: str) -> bool:
    """
    Loose stemming so 'dishwash' matches 'dishwasher', 'ship' matches
    'shipping', etc, without pulling in a full NLP stemmer dependency.
    Requires a shared prefix of at least _STEM_MIN_LEN characters so we
    don't get accidental matches on short/common words.
    """
    if query_term == chunk_term:
        return True
    shorter, longer = sorted([query_term, chunk_term], key=len)
    if len(shorter) < _STEM_MIN_LEN:
        return False
    return longer.startswith(shorter)


def _score_chunk(query_terms: list[str], chunk: Chunk) -> float:
    chunk_terms = _tokenize(chunk.text)
    if not chunk_terms:
        return 0.0
    chunk_term_counts: dict[str, int] = {}
    for t in chunk_terms:
        chunk_term_counts[t] = chunk_term_counts.get(t, 0) + 1

    score = 0.0
    for qt in query_terms:
        # exact match first (uses real IDF weight)
        if qt in chunk_term_counts:
            tf = chunk_term_counts[qt] / len(chunk_terms)
            idf = _IDF.get(qt, 1.0)
            score += tf * idf
            continue
        # fall back to stem match against distinct chunk terms, so
        # 'dishwash' still credits a chunk that only contains 'dishwasher'
        for ct, count in chunk_term_counts.items():
            if _stem_match(qt, ct):
                tf = count / len(chunk_terms)
                idf = _IDF.get(ct, 1.0)
                score += tf * idf * 0.8  # slight discount vs an exact match
                break

    # Small bonus if the query term appears in the heading -- headings are
    # a strong relevance signal ("Return window" heading for a return-window query).
    heading_terms = set(_tokenize(chunk.heading))
    heading_hits = sum(
        1 for qt in query_terms
        if qt in heading_terms or any(_stem_match(qt, ht) for ht in heading_terms)
    )
    score += heading_hits * 0.5

    # Title-match bonus: if a query term matches a word in the document's
    # own title (front-matter "title"), that's a strong topical signal --
    # e.g. a query mentioning "trailplus" should favor the TrailPlus doc
    # over a generically-worded "return window" section in another doc.
    title_terms = set(_tokenize(chunk.metadata.get("title", "")))
    title_hits = sum(
        1 for qt in query_terms
        if qt in title_terms or any(_stem_match(qt, tt) for tt in title_terms)
    )
    score += title_hits * 1.0

    # Authority weighting: demote superseded/draft/non-official content so
    # it doesn't outrank active official policy for the same keywords.
    # It is NOT excluded entirely -- it can still surface (e.g. for the
    # prompt-injection test case) but ranked lower and clearly labeled.
    if chunk.status == "superseded":
        score *= 0.35
    elif chunk.status == "draft":
        score *= 0.2
    if chunk.policy_authority == "none":
        score *= 0.2

    return score

# Small, explicit synonym map: customers describe symptoms ("broken zipper",
# "torn strap"), while policy docs describe categories ("damaged", "defective").
# This bridges that gap on the QUERY side only -- source documents are never
# modified, per the assignment's rules. Not exhaustive; documented as a known
# limitation (a production system would use semantic embeddings instead).
_SYNONYM_EXPANSIONS = {
    "broken": ["damaged", "defective"],
    "zipper": ["defective", "damaged"],
    "torn": ["damaged", "defective"],
    "ripped": ["damaged", "defective"],
    "cracked": ["damaged", "defective"],
    "leaking": ["damaged", "defective"],
    "faulty": ["defective", "damaged"],
    "wrong": ["incorrect"],
}


def _expand_query_terms(terms: list[str]) -> list[str]:
    expanded = list(terms)
    for t in terms:
        expanded.extend(_SYNONYM_EXPANSIONS.get(t, []))
    return expanded

def search_kb(query: str, top_k: int = 5) -> list[dict]:
    """
    Return the top_k most relevant chunks for `query`, each as a dict:
      {
        "doc_id": "...",
        "heading": "...",
        "text": "...",
        "score": float,
        "status": "active" | "superseded" | "draft",
        "policy_authority": "official" | "none",
        "source_label": "01-returns-policy-current.md > Standard return window",
      }

    Results are sorted by score descending. Callers (the agent's system
    prompt / tool-use logic) are responsible for deciding how to treat
    non-active or non-official chunks -- this function surfaces them
    with metadata rather than hiding them silently.
    """
    query_terms = _tokenize(query)
    if not query_terms:
        return []
    query_terms = _expand_query_terms(query_terms)

    scored = [(c, _score_chunk(query_terms, c)) for c in _ALL_CHUNKS]
    scored = [(c, s) for c, s in scored if s > 0]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    results = []
    for chunk, score in scored[:top_k]:
        results.append({
            "doc_id": chunk.doc_id,
            "heading": chunk.heading,
            "text": chunk.text,
            "score": round(score, 4),
            "status": chunk.status,
            "policy_authority": chunk.policy_authority,
            "source_label": chunk.source_label(),
        })
    return results


# ---- Manual smoke test ----------------------------------------------------

if __name__ == "__main__":
    test_queries = [
        "how long to return a backpack",
        "trailplus return window",
        "can I dishwash the breeze tumbler",
        "ship to Canada",
        "lifetime warranty",
    ]
    for q in test_queries:
        print(f"\n=== query: {q!r} ===")
        for r in search_kb(q, top_k=3):
            print(f"  [{r['score']:.3f}] {r['source_label']}  (status={r['status']}, authority={r['policy_authority']})")