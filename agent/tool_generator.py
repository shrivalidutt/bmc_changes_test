# ============================================================
#  tool_generator.py
#  Reads api_registry.yaml and auto-generates LangChain tools.
#  Supports:
#    - Bearer-token authentication (registry-defined login flow)
#    - In-memory token caching per Python process
#    - verify_ssl flag (for self-signed internal hosts)
#    - Parameter defaults from the registry
# ============================================================

import json
import os
from pathlib import Path
from typing import Any, Optional

import requests
import urllib3
import yaml
from langchain.tools import Tool

# Load .env from the agent/ directory at import time, so AUTOMATION_USER /
# AUTOMATION_PASS (and any future secrets) are available whether the caller
# is agent.py or tool_generator.py running standalone.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    # dotenv is optional — the env vars can still be set directly in the shell.
    pass


# ── Module-level token cache ──────────────────────────────────
# Keyed by the auth endpoint so multiple auth configs could coexist later.
_TOKEN_CACHE: dict[str, str] = {}


def _bmc_log(msg: str) -> None:
    """Print BMC call activity to the terminal (visible under npm start)."""
    print(msg, flush=True)


def log_bmc_auth_call(method: str, url: str, *, refresh: bool = False) -> None:
    action = "re-authenticate" if refresh else "authenticate"
    _bmc_log(f"\n[BMC API] {method} {url}  ({action})")


def log_bmc_api_call(
    api_def: dict,
    method: str,
    url: str,
    *,
    query_params: dict,
    body_params: dict,
    path_params: dict,
) -> None:
    lines = [
        "",
        "[BMC API] ────────────────────────────────────────────────",
        f"  API    : {api_def.get('id', '?')}",
        f"  Name   : {api_def.get('name', api_def.get('id', '?'))}",
        f"  Method : {method}",
        f"  URL    : {url}",
    ]
    if path_params:
        lines.append(f"  Path   : {json.dumps(path_params, ensure_ascii=False)}")
    if query_params:
        lines.append(f"  Query  : {json.dumps(query_params, ensure_ascii=False)}")
    if body_params:
        lines.append(f"  Body   : {json.dumps(body_params, ensure_ascii=False)}")
    _bmc_log("\n".join(lines))


def log_bmc_api_response(api_id: str, status_code: int) -> None:
    outcome = "OK" if status_code < 400 else "FAILED"
    _bmc_log(f"[BMC API] ← {api_id}  HTTP {status_code}  {outcome}\n")


def load_registry(yaml_path: str = "api_registry.yaml") -> dict:
    """Load the API registry from YAML."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_url(base_url: str, endpoint: str, path_params: dict) -> str:
    """Replace {param} placeholders in endpoint with actual values."""
    url = base_url.rstrip("/") + endpoint
    for key, value in path_params.items():
        url = url.replace(f"{{{key}}}", str(value))
    return url


# ── Auth ──────────────────────────────────────────────────────

def _resolve_credentials(auth_cfg: dict) -> tuple[str, str]:
    """Pull username/password from env vars named in the registry."""
    creds = auth_cfg.get("credentials", {}) or {}
    user_env = creds.get("username_env")
    pass_env = creds.get("password_env")
    if not user_env or not pass_env:
        raise RuntimeError(
            "auth.credentials.username_env / password_env must be set in "
            "api_registry.yaml"
        )
    user = os.environ.get(user_env)
    pwd = os.environ.get(pass_env)
    if not user or not pwd:
        raise RuntimeError(
            f"Missing credentials: set environment variables {user_env} and "
            f"{pass_env} before running the agent."
        )
    return user, pwd


def _extract_token(payload: Any, token_field: str) -> Optional[str]:
    """Find the token in the login response, tolerating a few shapes."""
    if isinstance(payload, dict):
        if token_field in payload and isinstance(payload[token_field], str):
            return payload[token_field]
        # Common nesting: { "session": { "token": "..." } } etc.
        for v in payload.values():
            if isinstance(v, dict):
                found = _extract_token(v, token_field)
                if found:
                    return found
    return None


def get_auth_token(
    auth_cfg: dict,
    base_url: str,
    verify_ssl: bool,
    force_refresh: bool = False,
) -> str:
    """Log in (if needed) and return a bearer token. Cached per endpoint."""
    endpoint = auth_cfg["endpoint"]
    cache_key = base_url.rstrip("/") + endpoint

    if not force_refresh and cache_key in _TOKEN_CACHE:
        return _TOKEN_CACHE[cache_key]

    user, pwd = _resolve_credentials(auth_cfg)
    url = build_url(base_url, endpoint, {})
    method = (auth_cfg.get("method") or "POST").upper()
    body = {"username": user, "password": pwd}

    log_bmc_auth_call(method, url, refresh=force_refresh)

    resp = requests.request(
        method,
        url,
        json=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=15,
        verify=verify_ssl,
    )
    log_bmc_api_response("session/login", resp.status_code)

    if resp.status_code >= 400:
        raise RuntimeError(
            f"Login failed ({resp.status_code}) at {url}: {resp.text[:300]}"
        )

    try:
        payload = resp.json()
    except ValueError:
        raise RuntimeError(f"Login response was not JSON: {resp.text[:300]}")

    token = _extract_token(payload, auth_cfg.get("token_field", "token"))
    if not token:
        raise RuntimeError(
            f"Could not find token field '{auth_cfg.get('token_field')}' in "
            f"login response: {payload}"
        )

    _TOKEN_CACHE[cache_key] = token
    return token


def _build_auth_header(auth_cfg: dict, token: str) -> dict:
    name = auth_cfg.get("header", "Authorization")
    fmt = auth_cfg.get("header_format", "Bearer {token}")
    return {name: fmt.format(token=token)}


# ── Per-API caller ────────────────────────────────────────────

def make_api_caller(api_def: dict, registry: dict):
    """
    Return a callable that executes the API call described by api_def.
    Input is a JSON string of parameters collected by the agent.
    Handles auth and parameter defaults automatically.
    """
    base_url = registry["base_url"]
    verify_ssl = bool(registry.get("verify_ssl", True))
    auth_cfg = registry.get("auth")

    if not verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def caller(params_json: str) -> str:
        try:
            if isinstance(params_json, str):
                params = json.loads(params_json) if params_json.strip() else {}
            else:
                params = params_json or {}
        except json.JSONDecodeError:
            return f"ERROR: Could not parse parameters. Expected JSON, got: {params_json}"

        # Split params by location + apply registry defaults for anything missing.
        path_params: dict = {}
        query_params: dict = {}
        body_params: dict = {}

        for p in api_def.get("parameters", []):
            name = p["name"]
            location = p.get("in", "query")
            if name in params and params[name] not in (None, ""):
                value = params[name]
            elif "default" in p:
                value = p["default"]
            elif p.get("required"):
                return (
                    f"ERROR: Missing required parameter '{name}'. "
                    "The agent must collect this before calling the API."
                )
            else:
                continue

            # Enforce enum at the HTTP boundary (case-insensitive match).
            if p.get("enum"):
                match = next(
                    (a for a in p["enum"] if str(a).lower() == str(value).strip().lower()),
                    None,
                )
                if match is None:
                    return (
                        f"ERROR: Invalid value for '{name}': {value!r}. "
                        f"Allowed values: {', '.join(str(a) for a in p['enum'])}."
                    )
                value = match

            if location == "path":
                path_params[name] = value
            elif location == "query":
                query_params[name] = value
            elif location == "body":
                body_params[name] = value

        url = build_url(base_url, api_def["endpoint"], path_params)
        method = api_def["method"].upper()

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Attach bearer token if this API requires auth.
        if api_def.get("requires_auth"):
            if not auth_cfg:
                return "ERROR: API requires auth but no `auth` block in registry."
            try:
                token = get_auth_token(auth_cfg, base_url, verify_ssl)
            except Exception as e:
                return f"ERROR: Authentication failed: {e}"
            headers.update(_build_auth_header(auth_cfg, token))

        def _do_request() -> requests.Response:
            return requests.request(
                method,
                url,
                params=query_params if method in ("GET", "DELETE") else query_params or None,
                json=body_params if method in ("POST", "PUT", "PATCH") else None,
                headers=headers,
                timeout=20,
                verify=verify_ssl,
            )

        try:
            log_bmc_api_call(
                api_def,
                method,
                url,
                query_params=query_params,
                body_params=body_params,
                path_params=path_params,
            )
            resp = _do_request()

            # If token expired mid-session, refresh once and retry.
            if api_def.get("requires_auth") and resp.status_code in (401, 403):
                try:
                    token = get_auth_token(auth_cfg, base_url, verify_ssl, force_refresh=True)
                    headers.update(_build_auth_header(auth_cfg, token))
                    _bmc_log(f"[BMC API] retrying {api_def.get('id', '?')} after token refresh")
                    resp = _do_request()
                except Exception as e:
                    return f"ERROR: Re-authentication failed: {e}"

            log_bmc_api_response(api_def.get("id", "?"), resp.status_code)

            try:
                data = resp.json() if resp.content else {}
            except ValueError:
                data = {"raw": resp.text}

            result = {
                "status_code": resp.status_code,
                "success": resp.status_code < 400,
                "data": data,
            }
            return json.dumps(result, indent=2)

        except requests.exceptions.ConnectionError:
            return f"ERROR: Could not connect to {base_url}. Check host/VPN/firewall."
        except requests.exceptions.Timeout:
            return "ERROR: The API request timed out."
        except Exception as e:
            return f"ERROR: Unexpected error calling API: {e}"

    return caller


def generate_tool_description(api_def: dict) -> str:
    """Rich tool description for the LLM, including defaults."""
    lines = [api_def["description"].strip(), ""]

    required_params = [p for p in api_def.get("parameters", []) if p.get("required")]
    optional_params = [p for p in api_def.get("parameters", []) if not p.get("required")]

    def _fmt(p: dict) -> str:
        parts = [f"  - {p['name']} ({p['type']}): {p['description'].strip()}"]
        if p.get("enum"):
            parts.append(f"    Allowed: {', '.join(str(v) for v in p['enum'])}")
        if "default" in p:
            parts.append(f"    Default: {p['default']}")
        return "\n".join(parts)

    if required_params:
        lines.append("REQUIRED parameters (must collect from user before calling):")
        lines.extend(_fmt(p) for p in required_params)

    if optional_params:
        lines.append("OPTIONAL parameters (ask user if they want to provide):")
        lines.extend(_fmt(p) for p in optional_params)

    lines.append("")
    lines.append("Pass all collected parameters as a JSON object string.")
    lines.append(f"HTTP Method: {api_def['method']}  |  Endpoint: {api_def['endpoint']}")
    if api_def.get("requires_auth"):
        lines.append("This call is authenticated — the agent handles login + bearer token automatically.")

    return "\n".join(lines)


def generate_tools(yaml_path: str = "api_registry.yaml") -> list[Tool]:
    """Read the YAML registry and return a list of LangChain Tool objects."""
    registry = load_registry(yaml_path)
    tools = []

    for api_def in registry["apis"]:
        tool = Tool(
            name=api_def["id"],
            description=generate_tool_description(api_def),
            func=make_api_caller(api_def, registry),
        )
        tools.append(tool)
        auth_tag = " 🔒" if api_def.get("requires_auth") else ""
        print(f"  ✅ Registered tool: [{api_def['method']}] {api_def['id']}{auth_tag}")

    print(f"\n  Total tools generated: {len(tools)}\n")
    return tools


# ── Standalone test ───────────────────────────────────────────
if __name__ == "__main__":
    here = Path(__file__).parent
    print("Loading tools from api_registry.yaml...\n")
    tools = generate_tools(str(here / "api_registry.yaml"))
    print("Tools loaded:")
    for t in tools:
        print(f"  - {t.name}")
