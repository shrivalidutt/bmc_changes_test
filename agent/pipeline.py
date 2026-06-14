"""
Explicit multi-step pipeline for automation chat.

Each step:
  - receives the same original user query (via PipelineContext)
  - receives structured output from prior steps (via PipelineContext)
  - produces structured JSON output stored on the context for the next step

One step runs per AutomationChatSession.run_pipeline_step() call.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from intent_router import classify_top_intent

from agent import (
    _drop_unmentioned_enums,
    _enforce_types,
    _optional_param_names,
    _required_param_names,
    apply_conversion_and_reconcile,
    apply_optional_defaults,
    format_param_ask,
    get_missing_params,
    latest_supplemental_line,
    phase1_detect_intent,
    phase2_extract_params,
    phase4_explain,
)


class PipelineStep(str, Enum):
    TOP_INTENT = "top_intent"
    SELECT_API = "select_api"
    EXTRACT_REQUIRED = "extract_required"
    EXTRACT_OPTIONAL = "extract_optional"
    CONVERT_PARAMS = "convert_params"
    EXECUTE = "execute"
    EXPLAIN = "explain"
    DONE = "done"


PIPELINE_ORDER = [
    PipelineStep.TOP_INTENT,
    PipelineStep.SELECT_API,
    PipelineStep.EXTRACT_REQUIRED,
    PipelineStep.EXTRACT_OPTIONAL,
    PipelineStep.CONVERT_PARAMS,
    PipelineStep.EXECUTE,
    PipelineStep.EXPLAIN,
    PipelineStep.DONE,
]


def next_step(current: PipelineStep) -> PipelineStep:
    idx = PIPELINE_ORDER.index(current)
    if idx + 1 >= len(PIPELINE_ORDER):
        return PipelineStep.DONE
    return PIPELINE_ORDER[idx + 1]


@dataclass
class PipelineContext:
    """Carries the user query and each step's structured output forward."""

    original_query: str
    supplemental_query: str = ""
    top_intent: dict = field(default_factory=dict)
    intent_result: dict = field(default_factory=dict)
    api_id: Optional[str] = None
    raw_params: dict = field(default_factory=dict)
    converted_params: dict = field(default_factory=dict)
    api_response: Optional[str] = None
    explanation: Optional[str] = None
    step_outputs: dict = field(default_factory=dict)

    @property
    def source_text(self) -> str:
        parts = [self.original_query.strip()]
        if self.supplemental_query.strip():
            parts.append(self.supplemental_query.strip())
        return "\n".join(p for p in parts if p)

    def record(self, step: PipelineStep, output: dict) -> None:
        self.step_outputs[step.value] = output


@dataclass
class PipelineDeps:
    apis: list
    api_map: dict
    api_catalog: str
    tool_map: dict
    intent_registry: dict
    history: list


@dataclass
class StepResult:
    step: PipelineStep
    output: dict
    next_step: PipelineStep
    user_message: Optional[str] = None
    waiting_for_user: bool = False
    pipeline_complete: bool = False
    error: Optional[str] = None


def _debug_step(step: PipelineStep, output: dict) -> None:
    if os.getenv("PIPELINE_DEBUG") or os.getenv("LLM_DEBUG"):
        print(f"\n[pipeline] {step.value} → {json.dumps(output, default=str)[:500]}\n", flush=True)


def run_pipeline_step(
    step: PipelineStep,
    ctx: PipelineContext,
    deps: PipelineDeps,
) -> StepResult:
    """Run exactly one pipeline step and return structured output for the next."""
    
    # Print human-readable progress so the user knows what's happening
    step_messages = {
        PipelineStep.TOP_INTENT: " Analyzing your request to figure out what you want to do...",
        PipelineStep.SELECT_API: " Finding the exact Control-M API for the job...",
        PipelineStep.EXTRACT_REQUIRED: " Reading your text to extract required parameters...",
        PipelineStep.EXTRACT_OPTIONAL: " Looking for any optional parameters...",
        PipelineStep.CONVERT_PARAMS: " Formatting the extracted parameters...",
        PipelineStep.EXECUTE: " Executing the API call to the server...",
        PipelineStep.EXPLAIN: " Summarizing the results...",
    }
    if step in step_messages:
        print(f"\n▶ {step_messages[step]}", flush=True)

    if step == PipelineStep.TOP_INTENT:
        return _step_top_intent(ctx, deps)
    if step == PipelineStep.SELECT_API:
        return _step_select_api(ctx, deps)
    if step == PipelineStep.EXTRACT_REQUIRED:
        return _step_extract_required(ctx, deps)
    if step == PipelineStep.EXTRACT_OPTIONAL:
        return _step_extract_optional(ctx, deps)
    if step == PipelineStep.CONVERT_PARAMS:
        return _step_convert_params(ctx, deps)
    if step == PipelineStep.EXECUTE:
        return _step_execute(ctx, deps)
    if step == PipelineStep.EXPLAIN:
        return _step_explain(ctx, deps)
    return StepResult(
        step=PipelineStep.DONE,
        output={},
        next_step=PipelineStep.DONE,
        pipeline_complete=True,
    )


def _step_top_intent(ctx: PipelineContext, deps: PipelineDeps) -> StepResult:
    top = classify_top_intent(
        ctx.original_query, deps.intent_registry, deps.apis, deps.history
    )
    ctx.top_intent = top
    ctx.record(PipelineStep.TOP_INTENT, top)
    _debug_step(PipelineStep.TOP_INTENT, top)

    if top["intent"] in ("chitchat", "help"):
        return StepResult(
            step=PipelineStep.TOP_INTENT,
            output=top,
            next_step=PipelineStep.DONE,
            user_message=top.get("reply"),
            pipeline_complete=True,
        )

    if top["intent"] == "faq_question":
        # pyrefly: ignore [missing-import]
        import faq_handler
        print("\n Consulting Control-M Documentation...")
        answer = faq_handler.get_answer(ctx.original_query)
        return StepResult(
            step=PipelineStep.TOP_INTENT,
            output=top,
            next_step=PipelineStep.DONE,
            user_message=answer,
            pipeline_complete=True,
        )

    return StepResult(
        step=PipelineStep.TOP_INTENT,
        output=top,
        next_step=PipelineStep.SELECT_API,
    )


def _step_select_api(ctx: PipelineContext, deps: PipelineDeps) -> StepResult:
    intent = phase1_detect_intent(
        ctx.original_query,
        deps.api_catalog,
        deps.history,
        set(deps.api_map),
        apis=deps.apis,
    )
    ctx.intent_result = intent
    api_id = intent.get("api_id")
    ctx.api_id = api_id if api_id in deps.api_map else None
    output = {
        "api_id": ctx.api_id,
        "confidence": intent.get("confidence"),
        "reason": intent.get("reason"),
        "reply": intent.get("reply"),
    }
    ctx.record(PipelineStep.SELECT_API, output)
    _debug_step(PipelineStep.SELECT_API, output)

    if not ctx.api_id:
        reply = intent.get("reply")
        if reply and str(reply).lower() != "null":
            msg = reply
        else:
            reason = intent.get("reason", "Could you be more specific?")
            msg = (
                f"I'm not sure which API fits that request. {reason}\n\n"
                "Try describing the automation task (e.g. 'list centralized "
                "connection profiles of type Database')."
            )
        return StepResult(
            step=PipelineStep.SELECT_API,
            output=output,
            next_step=PipelineStep.DONE,
            user_message=msg,
            pipeline_complete=True,
            error=intent.get("reason"),
        )

    return StepResult(
        step=PipelineStep.SELECT_API,
        output=output,
        next_step=PipelineStep.EXTRACT_REQUIRED,
    )


def _step_extract_required(ctx: PipelineContext, deps: PipelineDeps) -> StepResult:
    api = deps.api_map[ctx.api_id]
    req_names = _required_param_names(api)
    extracted: dict = {}
    is_followup = bool(ctx.supplemental_query.strip())
    strict_filters = not is_followup
    latest_line = latest_supplemental_line(ctx.supplemental_query) if is_followup else None

    if req_names:
        extracted = phase2_extract_params(
            ctx.source_text,
            api,
            ctx.raw_params,
            allowed_names=req_names,
            apply_confidence_filters=strict_filters,
            latest_followup_line=latest_line,
        )
        extracted = _drop_unmentioned_enums(api, extracted, ctx.source_text)
        ctx.raw_params.update(extracted)

    req_miss, _ = get_missing_params(api, ctx.raw_params)
    output = {"raw_params": dict(ctx.raw_params), "extracted": extracted}
    ctx.record(PipelineStep.EXTRACT_REQUIRED, output)
    _debug_step(PipelineStep.EXTRACT_REQUIRED, output)

    if req_miss:
        return StepResult(
            step=PipelineStep.EXTRACT_REQUIRED,
            output=output,
            next_step=PipelineStep.CONVERT_PARAMS,
            user_message=(
                "I still need a bit more information:\n\n"
                + format_param_ask(req_miss, [], include_optional=False)
            ),
            waiting_for_user=True,
        )

    return StepResult(
        step=PipelineStep.EXTRACT_REQUIRED,
        output=output,
        next_step=PipelineStep.EXTRACT_OPTIONAL,
    )


def _step_extract_optional(ctx: PipelineContext, deps: PipelineDeps) -> StepResult:
    api = deps.api_map[ctx.api_id]
    opt_names = _optional_param_names(api)
    extracted: dict = {}
    strict_filters = not ctx.supplemental_query.strip()

    if opt_names:
        extracted = phase2_extract_params(
            ctx.source_text,
            api,
            ctx.raw_params,
            allowed_names=opt_names,
            apply_confidence_filters=strict_filters,
        )
        extracted = _drop_unmentioned_enums(api, extracted, ctx.source_text)
        ctx.raw_params.update(extracted)

    apply_optional_defaults(api, ctx.raw_params)
    output = {"raw_params": dict(ctx.raw_params), "extracted": extracted}
    ctx.record(PipelineStep.EXTRACT_OPTIONAL, output)
    _debug_step(PipelineStep.EXTRACT_OPTIONAL, output)

    return StepResult(
        step=PipelineStep.EXTRACT_OPTIONAL,
        output=output,
        next_step=PipelineStep.CONVERT_PARAMS,
    )


def _step_convert_params(ctx: PipelineContext, deps: PipelineDeps) -> StepResult:
    api = deps.api_map[ctx.api_id]
    working = dict(ctx.raw_params)
    strict_filters = not ctx.supplemental_query.strip()
    converted, req_miss = apply_conversion_and_reconcile(
        api, working, ctx.source_text, apply_confidence_filters=strict_filters
    )
    ctx.raw_params = working
    ctx.converted_params = converted

    output = {"converted_params": dict(converted)}
    ctx.record(PipelineStep.CONVERT_PARAMS, output)
    _debug_step(PipelineStep.CONVERT_PARAMS, output)

    if req_miss:
        return StepResult(
            step=PipelineStep.CONVERT_PARAMS,
            output=output,
            next_step=PipelineStep.EXECUTE,
            user_message=(
                "I still need a bit more information:\n\n"
                + format_param_ask(req_miss, [], include_optional=False)
            ),
            waiting_for_user=True,
        )

    return StepResult(
        step=PipelineStep.CONVERT_PARAMS,
        output=output,
        next_step=PipelineStep.EXECUTE,
    )


def _step_execute(ctx: PipelineContext, deps: PipelineDeps) -> StepResult:
    api = deps.api_map[ctx.api_id]
    safe_params = _enforce_types(api, ctx.converted_params)
    try:
        raw = deps.tool_map[api["id"]].invoke(json.dumps(safe_params))
        ctx.api_response = raw
        output = {"api_response_preview": str(raw)[:500]}
        ctx.record(PipelineStep.EXECUTE, output)
        _debug_step(PipelineStep.EXECUTE, output)
        return StepResult(
            step=PipelineStep.EXECUTE,
            output=output,
            next_step=PipelineStep.EXPLAIN,
        )
    except Exception as exc:
        err_msg = f"I ran into an unexpected issue: {exc}"
        output = {"error": str(exc)}
        ctx.record(PipelineStep.EXECUTE, output)
        return StepResult(
            step=PipelineStep.EXECUTE,
            output=output,
            next_step=PipelineStep.DONE,
            user_message=err_msg,
            pipeline_complete=True,
            error=str(exc),
        )


def _step_explain(ctx: PipelineContext, deps: PipelineDeps) -> StepResult:
    api = deps.api_map[ctx.api_id]
    explanation = phase4_explain(api, ctx.api_response, ctx.original_query)
    ctx.explanation = explanation
    output = {"explanation": explanation}
    ctx.record(PipelineStep.EXPLAIN, output)
    _debug_step(PipelineStep.EXPLAIN, {"explanation_len": len(explanation)})

    return StepResult(
        step=PipelineStep.EXPLAIN,
        output=output,
        next_step=PipelineStep.DONE,
        user_message=explanation,
        pipeline_complete=True,
    )
