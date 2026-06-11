# BMC Automation Agent — LangChain + TinyLlama

A conversational AI agent for BMC Control-M automation APIs. Tools are auto-generated from `api_registry.yaml` and invoked through a local TinyLlama model.

---

## Project structure

```
agent/
├── api_registry.yaml     ← BMC API definitions (endpoints, params, auth)
├── intent_registry.yaml  ← Top-level routing: chitchat / help / tool_call
├── intent_router.py      ← Runs before API selection
├── tool_generator.py     ← YAML → LangChain tools
├── agent.py              ← Main agent (terminal CLI)
├── chat_session.py       ← Session + pipeline orchestration
├── pipeline.py           ← Step-by-step pipeline (one LLM call per step)
├── server.py             ← HTTP API for the UI chat widget
├── llm_provider.py       ← TinyLlama via Hugging Face
├── requirements.txt
└── .env.example          ← AUTOMATION_USER / AUTOMATION_PASS
```

---

## Setup (recommended — any OS)

From the **repo root**:

```bash
npm run install:agent
npm run start:agent
```

Edit `agent/.env` before starting (copy from `.env.example` if needed).

---

## Setup (manual)

**Windows (PowerShell or cmd):**

```bat
cd agent
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python server.py
```

**macOS / Linux:**

```bash
cd agent
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python server.py
```

CLI instead of HTTP: replace `server.py` with `agent.py`.

On first run, Hugging Face downloads `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (~2.2 GB).

List generated tools without chat:

```bash
# after install:agent
# Windows:  agent\.venv\Scripts\python benchmark.py --tools-only
# macOS:    agent/.venv/bin/python benchmark.py --tools-only
```

### Pipeline debug

Each user request runs through discrete steps (`top_intent` → `select_api` → `extract_required` → …). Set `PIPELINE_DEBUG=1` to log each step’s structured output. Set `PIPELINE_ONE_STEP=1` to run only one step per HTTP call (for testing).

Parameter extraction uses registry-derived confidence scoring (`param_confidence.py`). Tune with `PARAM_CONFIDENCE_THRESHOLD` (default `0.55`) — values that look like API topic words (e.g. “parameters” in “get agent parameters”) are rejected without hardcoded blocklists.

---

## Adding an API

Add an entry under `apis:` in `api_registry.yaml` — a new tool is generated automatically on restart.

---

## Tech stack

| Component | Technology |
|-----------|------------|
| LLM | TinyLlama 1.1B Chat (local) |
| Agent | Pipeline state machine (`pipeline.py` → `chat_session.py`) |
| Tools | Auto-generated from YAML |
| HTTP | Python `requests` + registry auth |
| Config | YAML + `.env` |
