"""
Registry-driven API selection helpers for Phase 1 (select_api).

Layers (in order inside phase1_detect_intent):
  1. Keyword/heuristic scoring from api_registry.yaml text
  2. TinyLlama JSON pick
  3. Normalize + fuzzy-map messy LLM output to the closest valid api_id
  4. Heuristic fallback when the LLM still fails
"""

from __future__ import annotations

import difflib
import re
from typing import Optional

# Words that appear in many queries/APIs — low signal for routing.
_GENERIC_TOKENS = frozenset({
    "want", "get", "list", "show", "find", "search", "look", "inspect",
    "with", "name", "type", "the", "for", "and", "from", "that", "this",
    "please", "give", "need", "use", "when", "user", "returns", "return",
    "server", "automation", "configured", "specified", "value", "field",
})

# Distinctive multi-word phrases → bonus toward APIs whose registry text matches.
_PHRASE_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("connection profile", ("connection", "profile")),
    ("connection profiles", ("connection", "profile")),
    ("agent param", ("agent", "parameter")),
    ("agent params", ("agent", "parameter")),
    ("agent parameter", ("agent", "parameter")),
    ("set agent", ("set", "agent", "parameter")),
    ("set parameter", ("set", "agent", "parameter")),
)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9_]{2,}", (text or "").lower()))


def _api_corpus(api: dict) -> str:
    parts = [
        api.get("id") or "",
        api.get("name") or "",
        api.get("description") or "",
        api.get("endpoint") or "",
        api.get("method") or "",
    ]
    return " ".join(str(p) for p in parts).lower()


def score_api_match(user_input: str, api: dict) -> float:
    """Score how well user text matches one registry API (higher = better)."""
    if not user_input or not user_input.strip():
        return 0.0

    norm = user_input.lower()
    user_tokens = _tokenize(norm)
    api_tokens = _tokenize(_api_corpus(api))
    if not user_tokens or not api_tokens:
        return 0.0

    shared = user_tokens & api_tokens
    signal = shared - _GENERIC_TOKENS
    score = float(len(signal))

    # Phrase hints (e.g. "connection profiles" → connection profile API).
    corpus = _api_corpus(api)
    for phrase, required_tokens in _PHRASE_HINTS:
        if phrase in norm and all(t in corpus for t in required_tokens):
            score += 4.0

    # Strong match when user text contains the exact registry id.
    api_id = api.get("id") or ""
    if api_id and api_id in norm:
        score += 6.0

    # Method-aware nudge: set/update language → POST APIs.
    if re.search(r"\b(set|update|change|configure)\b", norm):
        if (api.get("method") or "").upper() == "POST":
            score += 1.5
    elif re.search(r"\b(get|list|show|find|search|inspect)\b", norm):
        if (api.get("method") or "").upper() == "GET":
            score += 0.5

    return score


def match_api_heuristic(
    user_input: str,
    apis: list,
    *,
    min_score: float = 2.0,
    min_margin: float = 1.0,
) -> Optional[dict]:
    """
    Pick the best-matching API from registry text alone (no LLM).

    Returns phase1-shaped dict or None if ambiguous / no match.
    """
    if not user_input or not apis:
        return None

    ranked = sorted(
        ((score_api_match(user_input, api), api["id"]) for api in apis),
        reverse=True,
    )
    best_score, best_id = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0

    if best_score < min_score:
        return None
    if best_score - second_score < min_margin:
        return None

    return {
        "api_id": best_id,
        "confidence": "high" if best_score >= 4 else "medium",
        "reason": f"registry keyword match (score={best_score:.1f})",
        "reply": None,
    }


def fuzzy_match_api_id(candidate: str, valid_api_ids: set[str]) -> Optional[str]:
    """
    Map a near-miss api_id from sloppy LLM output to the closest registry id.

    Examples:
      get_connection_profiles → get_centralized_connection_profiles
      get_agent_param       → get_agent_parameters
    """
    if not candidate or not valid_api_ids:
        return None

    cand = str(candidate).strip().lower().replace("-", "_")
    if cand in valid_api_ids:
        return cand

    best_id: Optional[str] = None
    best_score = 0.0

    cand_tokens = [t for t in re.split(r"[_\s]+", cand) if len(t) > 2]

    for api_id in valid_api_ids:
        aid = api_id.lower()
        ratio = difflib.SequenceMatcher(None, cand, aid).ratio()

        # All significant tokens from candidate appear in registry id.
        if cand_tokens and all(t in aid for t in cand_tokens):
            ratio = max(ratio, 0.82)

        # Prefix/suffix overlap (get_*_connection_* style).
        if cand in aid or aid in cand:
            ratio = max(ratio, 0.78)

        if ratio > best_score:
            best_score = ratio
            best_id = api_id

    if best_id and best_score >= 0.62:
        return best_id
    return None


def normalize_intent_payload(
    text: str,
    valid_api_ids: set[str],
) -> Optional[dict]:
    """
    Turn messy SLM text into a phase1 JSON dict.

    Tries, in order:
      - strict / brace-sliced JSON
      - regex api_id extraction
      - registry id substring in text
      - fuzzy api_id from any quoted token or snake_case token in output
    """
    if not text:
        return None

    # --- strict JSON ---
    data = _try_parse_json(text)
    if data is not None:
        resolved = _resolve_api_id_field(data, valid_api_ids)
        if resolved is not None:
            return resolved

    # --- "api_id": "..." even outside valid JSON ---
    m = re.search(r'"api_id"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if m:
        resolved = _resolve_api_id_string(m.group(1), valid_api_ids)
        if resolved:
            return {
                "api_id": resolved,
                "confidence": "medium",
                "reason": "extracted api_id string from model output",
                "reply": None,
            }

    # --- api_id: value without quotes ---
    m = re.search(r'"api_id"\s*:\s*([a-zA-Z0-9_]+)', text, re.IGNORECASE)
    if m:
        resolved = _resolve_api_id_string(m.group(1), valid_api_ids)
        if resolved:
            return {
                "api_id": resolved,
                "confidence": "medium",
                "reason": "extracted unquoted api_id from model output",
                "reply": None,
            }

    # --- full registry id embedded in prose / echoed catalog ---
    for api_id in sorted(valid_api_ids, key=len, reverse=True):
        if api_id in text:
            return {
                "api_id": api_id,
                "confidence": "medium",
                "reason": "model output referenced registry api id",
                "reply": None,
            }

    # --- snake_case tokens that fuzzy-match a registry id ---
    for token in re.findall(r"\b[a-z][a-z0-9_]{5,}\b", text.lower()):
        resolved = fuzzy_match_api_id(token, valid_api_ids)
        if resolved:
            return {
                "api_id": resolved,
                "confidence": "low",
                "reason": f"fuzzy-mapped token '{token}' to registry api id",
                "reply": None,
            }

    return None


def _try_parse_json(text: str) -> Optional[dict]:
    import json

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        chunk = text[start : end + 1]
        try:
            obj = json.loads(chunk)
            return obj if isinstance(obj, dict) else None
        except Exception:
            # Common SLM mistake: trailing junk or single quotes.
            chunk = chunk.replace("'", '"')
            try:
                obj = json.loads(chunk)
                return obj if isinstance(obj, dict) else None
            except Exception:
                pass
    return None


def _resolve_api_id_string(raw: str, valid_api_ids: set[str]) -> Optional[str]:
    if not raw or str(raw).lower() in ("null", "none"):
        return None
    s = str(raw).strip()
    if s in valid_api_ids:
        return s
    return fuzzy_match_api_id(s, valid_api_ids)


def _resolve_api_id_field(data: dict, valid_api_ids: set[str]) -> Optional[dict]:
    api_id = data.get("api_id")
    if api_id is None or str(api_id).lower() in ("null", "none"):
        out = dict(data)
        out["api_id"] = None
        return out

    resolved = _resolve_api_id_string(str(api_id), valid_api_ids)
    if not resolved:
        return None

    out = dict(data)
    out["api_id"] = resolved
    if resolved != api_id:
        out["reason"] = (
            f"{data.get('reason') or 'llm pick'} "
            f"(fuzzy-mapped '{api_id}' → '{resolved}')"
        ).strip()
        out["confidence"] = data.get("confidence") or "low"
    return out
