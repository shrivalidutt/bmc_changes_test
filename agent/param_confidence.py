"""
Registry-driven confidence scoring for extracted parameter values.

Rejects values that echo the API's own id/name (including plural forms,
abbreviations like params/parameters, and close typos) instead of real
user-supplied identifiers. Works for any API in api_registry.yaml.

Set PARAM_CONFIDENCE_THRESHOLD (default 0.55) to tune acceptance.
"""

from __future__ import annotations

import difflib
import os
import re
from functools import lru_cache
from typing import Optional

# Shared with api_router._GENERIC_TOKENS — weak routing/extraction signals.
_GENERIC = frozenset({
    "want", "get", "list", "show", "find", "search", "look", "inspect",
    "with", "the", "for", "and", "from", "that", "this", "please", "give",
    "need", "use", "when", "user", "returns", "return", "configured",
    "specified", "connected", "following", "provide", "bit", "more",
    "information", "still", "automation", "control", "server", "agent",
})

# Intent verbs echoed from the user's request ("get agent parameters") — not param values.
_INTENT_VERBS = frozenset({
    "get", "list", "show", "find", "search", "look", "inspect", "set",
    "update", "change", "configure", "add", "give", "want", "need", "use",
    "audit", "returns", "return",
})

# Patterns with explicit assignment (group 1 = value) — index in extract_params_heuristic.
_EXPLICIT_PATTERN_COUNT = 7  # first 7 patterns use as/is/=/with

# Fuzzy match against API name tokens (params ↔ parameters, minor typos).
_NAME_ECHO_FUZZY_RATIO = 0.82


def param_confidence_threshold() -> float:
    raw = os.getenv("PARAM_CONFIDENCE_THRESHOLD", "0.55")
    try:
        return float(raw)
    except ValueError:
        return 0.55


def _base_api_name_tokens(api: dict) -> frozenset[str]:
    """
    Raw tokens from API id, display name, and static endpoint path segments.
    Includes words that are also parameter names (e.g. agent in get_agent_parameters)
    so intent phrasing is not mistaken for a supplied value.
    """
    tokens: set[str] = set()
    for part in re.split(r"[_\s]+", (api.get("id") or "").lower()):
        if len(part) > 1:
            tokens.add(part)
    for token in re.findall(r"[a-z][a-z0-9_]{2,}", (api.get("name") or "").lower()):
        tokens.add(token)
    endpoint = api.get("endpoint") or ""
    for seg in re.findall(r"[a-z][a-z0-9_]{2,}", endpoint.lower()):
        if "{" not in seg:
            tokens.add(seg)
    return frozenset(tokens)


def _expand_token_forms(token: str) -> set[str]:
    """
    Plural/singular and common truncations (params ↔ parameters) — no per-API rules.
    """
    t = token.lower().strip()
    if not t:
        return set()
    out = {t}
    if t.endswith("s") and len(t) > 3:
        out.add(t[:-1])
    else:
        out.add(t + "s")
    for n in (4, 5, 6):
        if len(t) > n:
            out.add(t[:n])
            out.add(t[:n] + "s")
    return out


@lru_cache(maxsize=256)
def _expanded_name_vocab(api_id: str, api_name: str, endpoint: str) -> frozenset[str]:
    api = {"id": api_id, "name": api_name, "endpoint": endpoint}
    expanded: set[str] = set()
    for token in _base_api_name_tokens(api):
        expanded.update(_expand_token_forms(token))
    return frozenset(expanded)


def build_api_name_vocabulary(api: dict) -> frozenset[str]:
    """
    All forms derived from this API's id, name, and endpoint path — intent words,
    not user identifiers. Includes params/parameters-style variants.
    """
    return _expanded_name_vocab(
        api.get("id") or "",
        api.get("name") or "",
        api.get("endpoint") or "",
    )


def _tokens_similar(a: str, b: str) -> bool:
    """Direct, morphological, prefix, or fuzzy match between two tokens."""
    if a == b:
        return True
    if _expand_token_forms(a) & _expand_token_forms(b):
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 4 and longer.startswith(shorter[:4]):
        return True
    if len(shorter) >= 3 and longer.startswith(shorter):
        if len(shorter) / len(longer) >= 0.45:
            return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= _NAME_ECHO_FUZZY_RATIO


def value_echoes_api_name(value: str, api: dict) -> bool:
    """
    True when `value` is the API name/id vocabulary (or a close variant/typo),
    e.g. parameters/params/paramter for get_agent_parameters.
    """
    low = str(value).strip().lower()
    if not low or len(low) < 2:
        return False
    if low in build_api_name_vocabulary(api):
        return True
    for token in _base_api_name_tokens(api):
        if _tokens_similar(low, token):
            return True
    return False


@lru_cache(maxsize=64)
def _vocab_key(api_id: str, corpus: str, param_names: tuple[str, ...]) -> frozenset[str]:
    tokens = set(re.findall(r"[a-z][a-z0-9_]{2,}", corpus.lower()))
    tokens -= _GENERIC
    for name in param_names:
        n = name.lower()
        tokens.add(n)
        if n.endswith("s") and len(n) > 3:
            tokens.add(n[:-1])
        elif not n.endswith("s"):
            tokens.add(n + "s")
    return frozenset(tokens)


def build_api_domain_vocabulary(api: dict) -> frozenset[str]:
    """
    Words that describe this API in the registry — not user-supplied identifiers.
    Built from id, name, description, endpoint, and sibling parameter names.
    """
    corpus = " ".join(
        str(api.get(k) or "")
        for k in ("id", "name", "description", "endpoint", "method")
    )
    param_names = tuple(p["name"] for p in api.get("parameters", []))
    return _vocab_key(api.get("id") or corpus[:32], corpus, param_names)


def build_api_identity_tokens(api: dict) -> frozenset[str]:
    """Alias for base name tokens (backward compatible)."""
    return _base_api_name_tokens(api)


def _looks_like_identifier(value: str) -> bool:
    return bool(
        re.search(r"\d", value)
        or re.search(r"[A-Z]", value)
        or "_" in value
        or "*" in value
        or "?" in value
    )


def _explicit_assignment_patterns(pname: str, low: str) -> list[str]:
    return [
        rf"\b{re.escape(pname)}\s+(?:as|is|to|=|:)\s*['\"]?{re.escape(low)}['\"]?",
        rf"\bwith\s+{re.escape(pname)}\s+(?:as\s+)?['\"]?{re.escape(low)}['\"]?",
        rf"\b{re.escape(pname)}\s*=\s*['\"]?{re.escape(low)}['\"]?",
    ]


def _has_explicit_assignment(
    pname: str,
    low: str,
    sval: str,
    text: str,
    *,
    pattern_index: Optional[int] = None,
) -> bool:
    if pattern_index is not None and pattern_index < _EXPLICIT_PATTERN_COUNT:
        return True
    if any(re.search(p, text, re.IGNORECASE) for p in _explicit_assignment_patterns(pname, low)):
        return True
    if _looks_like_identifier(sval):
        grounded = [
            rf"\b{re.escape(pname)}\s+{re.escape(low)}\b",
            rf"\bfor\s+{re.escape(pname)}\s+{re.escape(low)}\b",
        ]
        if any(re.search(p, text, re.IGNORECASE) for p in grounded):
            return True
    return False


def score_param_extraction_confidence(
    param_def: dict,
    value,
    api: dict,
    user_input: str,
    *,
    pattern_index: Optional[int] = None,
) -> float:
    """
    Return 0.0–1.0 confidence that `value` is a real user-supplied parameter
    value rather than API/topic vocabulary echoed from the request sentence.
    """
    if value is None or value == "":
        return 0.0

    sval = str(value).strip()
    if not sval:
        return 0.0

    low = sval.lower()
    pname = param_def["name"]
    text = (user_input or "").lower()
    vocab = build_api_domain_vocabulary(api)
    explicit = _has_explicit_assignment(
        pname, low, sval, text, pattern_index=pattern_index
    )
    enum_match = bool(
        param_def.get("enum")
        and any(str(a).lower() == low for a in param_def["enum"])
        and low in text
    )
    name_echo = value_echoes_api_name(sval, api)
    score = 0.55  # neutral baseline

    if explicit:
        score += 0.35
    elif pattern_index is not None and pattern_index >= _EXPLICIT_PATTERN_COUNT:
        score -= 0.25  # bare "agent parameters" style

    # Intent verb or API name/id echo (params, parameters, typos) — ask user instead.
    if not explicit and not enum_match:
        if low in _INTENT_VERBS:
            score -= 0.5
        if name_echo:
            score -= 0.55

    if low in vocab and not enum_match:
        score -= 0.55

    sibling_names = {p["name"].lower() for p in api.get("parameters", [])}
    if low in sibling_names and low != pname.lower():
        score -= 0.4

    if low not in text:
        score -= 0.7
    else:
        score += 0.1

    if _looks_like_identifier(sval):
        score += 0.15
    elif re.fullmatch(r"[a-z]{1,2}", low):
        if not explicit:
            score -= 0.15
        else:
            score += 0.1

    if enum_match:
        score += 0.3

    return max(0.0, min(1.0, score))


def accept_param_value(
    param_def: dict,
    value,
    api: dict,
    user_input: str,
    *,
    pattern_index: Optional[int] = None,
    threshold: Optional[float] = None,
) -> bool:
    thresh = param_confidence_threshold() if threshold is None else threshold
    return (
        score_param_extraction_confidence(
            param_def, value, api, user_input, pattern_index=pattern_index
        )
        >= thresh
    )
