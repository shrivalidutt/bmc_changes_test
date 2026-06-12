"""
Top-level intent routing (chitchat / help / tool_call) before API selection.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from agent import build_api_catalog, llm_intent, safe_json_parse

BASE_DIR = Path(__file__).parent
INTENT_REGISTRY_PATH = str(BASE_DIR / "intent_registry.yaml")

VALID_TOP_INTENTS = frozenset({"chitchat", "help", "tool_call", "faq_question"})


def load_intent_registry():
    with open(INTENT_REGISTRY_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _intent_map(registry) -> dict[str, dict]:
    return {i["id"]: i for i in registry.get("intents", [])}


def match_intent_heuristic(user_input: str, registry) -> str | None:
    """Fast path: exact / short-message match against registry triggers."""
    norm = _normalize(user_input)
    if not norm:
        return None

    for intent in registry.get("intents", []):
        iid = intent.get("id")
        if iid == "tool_call":
            continue
        for trigger in intent.get("triggers", []):
            t = _normalize(trigger)
            if not t:
                continue
            if norm == t or norm.startswith(t + " ") or norm.endswith(" " + t):
                return iid

    # Very short non-technical utterances → chitchat (avoid "agent" API false positives)
    if len(norm.split()) <= 2 and not _looks_like_automation_request(norm):
        if re.fullmatch(r"(hi+|hey+|hello+|yo+|sup|thanks?|thx|bye+)", norm.replace(" ", "")):
            return "chitchat"

    return None


def _looks_like_automation_request(norm: str) -> bool:
    keywords = (
        "profile", "connection", "parameter", "server", "config",
        "list", "get", "set", "show", "find", "search", "deploy",
        "centralized", "database", "agent param", "control-m", "controlm",
        "analyze", "analysis", "communication", "agentless", "desired",
        "state", "recycle", "host", "check", "test", "inspect", "audit",
    )
    return any(k in norm for k in keywords)


def _registry_matches_api(user_input: str, apis: list) -> bool:
    """True when api_registry keyword routing would pick an API (any registry size)."""
    if not user_input or not apis:
        return False
    from api_router import match_api_heuristic

    return match_api_heuristic(user_input, apis) is not None


def build_api_name_list(apis) -> list[str]:
    """Display names from api_registry.yaml (falls back to id)."""
    names: list[str] = []
    for api in apis or []:
        label = (api.get("name") or api.get("id") or "").strip()
        if label:
            names.append(label)
    return names


def build_chitchat_reply(apis) -> str:
    """Greeting built from registry API names only (plain ASCII)."""
    names = build_api_name_list(apis)
    if not names:
        return (
            "Hello! I am your Control-M automation assistant. "
            "Tell me what automation task you would like to run."
        )
    lines = [
        "Hello! I am your Control-M automation assistant. "
        "I can run registered BMC APIs for you:",
        "",
        *([f"- {name}" for name in names]),
        "",
        "What would you like to do?",
    ]
    return "\n".join(lines)


def build_help_reply(apis) -> str:
    table_lines = [
        "| Service | Description |",
        "| :--- | :--- |"
    ]
    for api in apis:
        desc = api.get("description", "").strip()
        first_sentence = desc.split(".")[0] + "." if desc else ""
        table_lines.append(f"| **{api['name']}** | {first_sentence} |")
    
    table_str = "\n".join(table_lines)
    return (
        "I can run these BMC automation APIs for you:\n\n"
        f"{table_str}\n\n"
        "Describe what you need — for example:\n"
        '• "List centralized connection profiles of type Database"\n'
        '• "Get parameters for server PROD and agent AG001"\n'
        '• "Set agent parameter X to value Y"\n\n'
        "I'll confirm the API and parameters before calling anything."
    )


def classify_top_intent(user_input: str, registry, apis, history) -> dict:
    """
    Returns {"intent": "chitchat"|"help"|"tool_call", "reply": "..."}.
    Heuristic first, then LLM for ambiguous messages.
    """
    imap = _intent_map(registry)
    hit = match_intent_heuristic(user_input, registry)
    if hit == "chitchat":
        return {"intent": "chitchat", "reply": build_chitchat_reply(apis)}
    if hit == "help":
        return {"intent": "help", "reply": build_help_reply(apis)}

    if hit == "faq_question":
        return {"intent": "faq_question", "reply": None}

    norm = _normalize(user_input)
    if _registry_matches_api(user_input, apis):
        return {"intent": "tool_call", "reply": None}
    if _looks_like_automation_request(norm):
        return {"intent": "tool_call", "reply": None}

    # LLM fallback for ambiguous text
    intent_lines = []
    for intent in registry.get("intents", []):
        intent_lines.append(f"- {intent['id']}: {intent.get('description', '').strip()}")

    history_ctx = ""
    if history:
        recent = history[-4:]
        history_ctx = "Recent conversation:\n" + "\n".join(
            f"  {m['role'].upper()}: {m['content'][:200]}" for m in recent
        ) + "\n\n"

    prompt = f"""{history_ctx}Classify the user's message into exactly ONE top-level intent.

Intents:
{chr(10).join(intent_lines)}

Rules:
- chitchat: greetings, thanks, small talk ONLY — no automation task
- help: asks what you can do / available commands
- faq_question: general questions about how things work, domain knowledge
- tool_call: wants to run a BMC automation action (list, get, set, search, configure)

User said:
<user_input>
{user_input}
</user_input>

Respond with ONLY JSON:
{{"intent": "chitchat"|"help"|"faq_question"|"tool_call", "reply": "<short friendly reply if chitchat, else null>"}}"""

    raw = llm_intent.invoke(prompt).content
    data = safe_json_parse(raw) or {}
    intent = data.get("intent")
    if intent not in VALID_TOP_INTENTS:
        intent = (
            "tool_call"
            if _registry_matches_api(user_input, apis)
            or _looks_like_automation_request(norm)
            else "chitchat"
        )

    reply = data.get("reply")
    if intent == "chitchat":
        reply = build_chitchat_reply(apis)
    elif intent == "help":
        reply = build_help_reply(apis)
    else:
        reply = None

    return {"intent": intent, "reply": reply}


def is_chitchat_message(user_input: str, registry) -> bool:
    return match_intent_heuristic(user_input, registry) == "chitchat"
