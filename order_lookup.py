"""
order_lookup.py

A single-purpose function: given a (possibly messy) order ID typed by a
customer, return ONLY the customer-safe fields for that order, or a
clear "not found" signal.

Design rules (from data/orders-data-dictionary.md):
- Normalize harmless input differences (case, whitespace) but do NOT
  guess a different order ID if there's no exact match after normalizing.
- Never include customer.name, customer.email, customer.shipping_address,
  or anything under "internal" (risk_score, warehouse_note, support_tags).
  These fields are stripped out in Python, BEFORE anything is returned --
  they should never even reach the model's context window.
- The "status" field is authoritative. Stale carrier/tracking/ETA fields
  on cancelled/returned orders are still returned as raw data here; it's
  the agent's job (system prompt) to not misuse them. This function's job
  is only to prevent PII/internal leakage and safe field selection.
"""

import json
import re
from pathlib import Path
from typing import Optional, TypedDict


# ---- Data loading -----------------------------------------------------

DATA_PATH = Path(__file__).parent / "data" / "orders.json"


def _load_orders() -> dict:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# Loaded once at import time. Fine for this assignment's scale (mock,
# read-only dataset). In a real system this would be a DB call per lookup.
_DATASET = _load_orders()
_ORDERS_BY_ID = {o["order_id"]: o for o in _DATASET["orders"]}
SNAPSHOT_AT = _DATASET["snapshot_at"]


# ---- Input normalization -----------------------------------------------

def normalize_order_id(raw: str) -> str:
    """
    Normalize harmless differences only:
      - strip surrounding whitespace
      - uppercase
    Does NOT try to fix typos or guess a close match -- per the data
    dictionary, guessing a substantially different ID is explicitly
    disallowed.
    """
    if raw is None:
        return ""
    cleaned = raw.strip().upper()
    return cleaned


_ORDER_ID_PATTERN = re.compile(r"^ORD-\d+$")


def looks_like_order_id(raw: str) -> bool:
    """Cheap shape check, used only to give a clearer 'malformed' message."""
    return bool(_ORDER_ID_PATTERN.match(normalize_order_id(raw)))


# ---- The customer-safe field allowlist ---------------------------------
# Copied 1:1 from orders-data-dictionary.md "Customer-safe fields".
# This is an ALLOWLIST, not a denylist -- new fields added to orders.json
# in the future are excluded by default unless explicitly added here.
# That's a deliberate safety choice: fail closed, not open.

_SAFE_TOP_LEVEL_FIELDS = [
    "order_id",
    "membership_tier",
    "placed_at",
    "status",
    "status_updated_at",
    "shipped_at",
    "delivered_at",
    "carrier",
    "tracking_number",
    "estimated_delivery",
    "customer_safe_message",
]

_SAFE_ITEM_FIELDS = ["name", "quantity", "final_sale"]


class LookupResult(TypedDict, total=False):
    found: bool
    order: Optional[dict]
    error: Optional[str]


def order_lookup(raw_order_id: str) -> LookupResult:
    """
    The only function that touches the raw orders dataset.

    Returns a dict that is SAFE to hand to the model:
      { "found": True,  "order": {...safe fields only...} }
      { "found": False, "error": "not_found" | "malformed" }

    Never returns customer PII or the "internal" block, under any
    circumstance -- even if asked. There is no parameter that can make
    this function return those fields.
    """
    if not raw_order_id or not raw_order_id.strip():
        return {"found": False, "error": "missing"}

    normalized = normalize_order_id(raw_order_id)

    if not looks_like_order_id(normalized):
        return {"found": False, "error": "malformed"}

    order = _ORDERS_BY_ID.get(normalized)
    if order is None:
        return {"found": False, "error": "not_found"}

    safe_order = {k: order.get(k) for k in _SAFE_TOP_LEVEL_FIELDS}
    safe_order["items"] = [
        {k: item.get(k) for k in _SAFE_ITEM_FIELDS}
        for item in order.get("items", [])
    ]

    return {"found": True, "order": safe_order}


# ---- Manual smoke test when run directly -------------------------------

if __name__ == "__main__":
    test_ids = [
        "ORD-1007",       # normal
        "  ord-1007  ",   # messy casing/whitespace -> should normalize
        "ORD-9999",       # unknown
        "not-an-id",      # malformed
        "",                # missing
        "ORD-1004",       # cancelled, stale fields present
        "ORD-1011",       # shipped, no ETA
    ]
    for tid in test_ids:
        result = order_lookup(tid)
        print(f"\ninput={tid!r}")
        print(json.dumps(result, indent=2))