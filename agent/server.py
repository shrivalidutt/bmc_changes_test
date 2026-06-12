#!/usr/bin/env python3
"""
HTTP API for the BMC chat widget.

POST /api/chat  { "message": "...", "session_id": "...", "continue_pipeline": false }
→ { "response": "...", "pipeline_complete": true, "waiting_for_user": false, "last_step": "..." }

Each pipeline step runs as a separate invocation; the widget auto-continues until
pipeline_complete or waiting_for_user.
"""

from __future__ import annotations

import os
import sys
import uuid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

from chat_session import store

load_dotenv()

app = Flask(__name__)
CORS(
    app,
    resources={r"/api/*": {"origins": os.getenv("CORS_ORIGINS", "*")}},
)

store.ensure_warmup()


@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "bmc-automation-agent"})


def _run_pipeline(session, message: str, continue_pipeline: bool):
    """Chain pipeline steps; each iteration is exactly one step."""
    result = session.handle(message, continue_pipeline=continue_pipeline)
    one_step = os.getenv("PIPELINE_ONE_STEP", "").strip().lower() in ("1", "true", "yes")
    if one_step:
        return result
    while (
        not result.pipeline_complete
        and not result.waiting_for_user
        and session.state == "PIPELINE"
    ):
        result = session.handle("", continue_pipeline=True)
    return result


@app.post("/api/chat")
def api_chat():
    body = request.get_json(silent=True) or {}
    message = (body.get("message") or "").strip()
    continue_pipeline = bool(body.get("continue_pipeline"))
    session_id = (body.get("session_id") or "").strip() or str(uuid.uuid4())

    if not message and not continue_pipeline:
        return jsonify({"error": "message is required"}), 400

    session = store.get(session_id)
    try:
        result = _run_pipeline(session, message, continue_pipeline)
        response_text = "\n\n".join(result.messages)
    except Exception as exc:
        app.logger.exception("chat error session=%s", session_id)
        return jsonify(
            {
                "session_id": session_id,
                "response": f"Something went wrong on the agent: {exc}",
                "pipeline_complete": True,
                "waiting_for_user": False,
            }
        ), 500

    payload = {
        "session_id": session_id,
        "response": response_text,
        "pipeline_complete": result.pipeline_complete,
        "waiting_for_user": result.waiting_for_user,
        "last_step": result.last_step,
    }
    if os.getenv("PIPELINE_DEBUG"):
        payload["step_outputs"] = result.step_outputs

    return jsonify(payload)


@app.post("/api/chat/reset")
def api_chat_reset():
    body = request.get_json(silent=True) or {}
    session_id = (body.get("session_id") or "").strip()
    if session_id:
        store.reset(session_id)
    return jsonify({"ok": True})


if __name__ == "__main__":
    # Default 5001 — macOS often reserves 5000 for AirPlay Receiver
    port = int(os.getenv("AGENT_PORT", "5001"))
    print(f"\nBMC Automation Agent API on http://localhost:{port}/api/chat\n")
    if port == 5000:
        print(
            "Tip: If this port fails, set AGENT_PORT=5001 (AirPlay uses 5000 on macOS).\n"
        )
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
