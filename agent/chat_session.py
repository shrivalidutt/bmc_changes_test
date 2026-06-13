"""
Registry-driven chat session for CLI and HTTP.

Pipeline model: each LLM/API step runs individually; structured output from
step N is stored on PipelineContext and passed into step N+1. The same user
query travels through every step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import faq_handler
from intent_router import classify_top_intent, load_intent_registry, match_intent_heuristic

from pipeline import (
    PipelineContext,
    PipelineDeps,
    PipelineStep,
    StepResult,
    run_pipeline_step,
)
from tool_generator import generate_tools

from agent import (
    REGISTRY_PATH,
    _drop_unmentioned_enums,
    _required_param_names,
    build_api_catalog,
    load_registry,
    phase2_extract_params,
    warmup_llm,
)


@dataclass
class HandleResult:
    """Result of one or more pipeline step invocations."""

    messages: list[str] = field(default_factory=list)
    pipeline_complete: bool = True
    waiting_for_user: bool = False
    last_step: Optional[str] = None
    step_outputs: dict = field(default_factory=dict)


class AutomationChatSession:
    """One conversation thread backed by api_registry.yaml tools."""

    def __init__(self):
        registry = load_registry()
        self.intent_registry = load_intent_registry()
        self.apis = registry["apis"]
        tools = generate_tools(REGISTRY_PATH)
        self.tool_map = {t.name: t for t in tools}
        self.api_map = {a["id"]: a for a in self.apis}
        self.api_catalog = build_api_catalog(self.apis)

        self.history: list[dict] = []
        self.state = "IDLE"
        self.pipeline_ctx: Optional[PipelineContext] = None
        self.pipeline_step: Optional[PipelineStep] = None
        self._replies: list[str] = []

    @property
    def _deps(self) -> PipelineDeps:
        return PipelineDeps(
            apis=self.apis,
            api_map=self.api_map,
            api_catalog=self.api_catalog,
            tool_map=self.tool_map,
            intent_registry=self.intent_registry,
            history=self.history,
        )

    def reset(self):
        self.state = "IDLE"
        self.pipeline_ctx = None
        self.pipeline_step = None

    def _say(self, msg: str, *, prefix: str = ""):
        text = f"{prefix}{msg}" if prefix else msg
        self._replies.append(text)
        self.history.append({"role": "assistant", "content": msg})

    def _start_pipeline(self, user_query: str) -> None:
        self.pipeline_ctx = PipelineContext(original_query=user_query)
        self.pipeline_step = PipelineStep.TOP_INTENT
        self.state = "PIPELINE"

    def run_one_pipeline_step(self) -> StepResult:
        """Advance the pipeline by exactly one step."""
        if not self.pipeline_ctx or not self.pipeline_step:
            raise RuntimeError("No active pipeline")

        result = run_pipeline_step(self.pipeline_step, self.pipeline_ctx, self._deps)
        self.pipeline_step = result.next_step

        if result.pipeline_complete:
            self.reset()
        elif result.waiting_for_user:
            self.state = "COLLECT_REQUIRED"
            # Re-run the same step after the user supplies missing required params.
            self.pipeline_step = result.step

        return result

    def handle(
        self,
        user_input: str,
        *,
        continue_pipeline: bool = False,
    ) -> HandleResult:
        """
        Process input and run exactly one pipeline step.

        - Normal user message in IDLE starts the pipeline at TOP_INTENT.
        - continue_pipeline=True advances an in-flight pipeline (no new query).
        - COLLECT_REQUIRED merges the follow-up then resumes at EXTRACT_REQUIRED.
        """
        self._replies = []
        text = (user_input or "").strip()
        handle_result = HandleResult()

        if continue_pipeline:
            if self.state != "PIPELINE" or not self.pipeline_step:
                return handle_result
            step = self.run_one_pipeline_step()
            self._apply_step_result(step, handle_result)
            return handle_result

        if not text:
            return handle_result

        low = text.lower()
        if low in ("exit", "quit", "bye"):
            self._say("Goodbye!")
            handle_result.messages = list(self._replies)
            return handle_result

        self.history.append({"role": "user", "content": text})

        if low in ("start over", "reset", "cancel", "nevermind", "new"):
            self.reset()
            self._say("No problem! What would you like to do?")
            handle_result.messages = list(self._replies)
            return handle_result

        if self.state != "IDLE":
            top_hit = match_intent_heuristic(text, self.intent_registry)
            if top_hit in ("chitchat", "help"):
                self.reset()
                top = classify_top_intent(
                    text, self.intent_registry, self.apis, self.history
                )
                self._say(top["reply"])
                handle_result.messages = list(self._replies)
                return handle_result

        if self.state == "IDLE":
            self._start_pipeline(text)
            step = self.run_one_pipeline_step()
            self._apply_step_result(step, handle_result)
            return handle_result

        if self.state == "COLLECT_REQUIRED":
            self._resume_from_collect(text)
            step = self.run_one_pipeline_step()
            self._apply_step_result(step, handle_result)
            return handle_result

        if self.state == "PIPELINE":
            self._say("I am still processing your previous request. Please wait a moment.")
            handle_result.messages = list(self._replies)
            handle_result.pipeline_complete = False
            return handle_result

        return handle_result

    def _resume_from_collect(self, user_input: str) -> None:
        """Merge follow-up required params, then re-run extraction before convert."""
        if not self.pipeline_ctx or not self.pipeline_ctx.api_id:
            self.reset()
            return

        prior = self.pipeline_ctx.supplemental_query.strip()
        self.pipeline_ctx.supplemental_query = (
            f"{prior}\n{user_input}".strip() if prior else user_input
        )
        api = self.api_map[self.pipeline_ctx.api_id]
        req_names = _required_param_names(api)
        if req_names:
            extracted = phase2_extract_params(
                self.pipeline_ctx.source_text,
                api,
                self.pipeline_ctx.raw_params,
                allowed_names=req_names,
                apply_confidence_filters=False,
                latest_followup_line=user_input,
            )
            extracted = _drop_unmentioned_enums(
                api, extracted, self.pipeline_ctx.source_text
            )
            self.pipeline_ctx.raw_params.update(extracted)

        self.state = "PIPELINE"
        self.pipeline_step = PipelineStep.EXTRACT_REQUIRED

    def _apply_step_result(self, step: StepResult, handle_result: HandleResult) -> None:
        if step.user_message:
            self._say(step.user_message)

        handle_result.last_step = step.step.value
        handle_result.pipeline_complete = step.pipeline_complete
        handle_result.waiting_for_user = step.waiting_for_user
        if self.pipeline_ctx:
            handle_result.step_outputs = dict(self.pipeline_ctx.step_outputs)
        handle_result.messages = list(self._replies)


class SessionStore:
    """In-memory sessions keyed by widget session_id."""

    def __init__(self):
        self._sessions: dict[str, AutomationChatSession] = {}
        self._warmed = False

    def ensure_warmup(self):
        if not self._warmed:
            warmup_llm()
            self._warmed = True

    def get(self, session_id: str) -> AutomationChatSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = AutomationChatSession()
        return self._sessions[session_id]

    def reset(self, session_id: str):
        self._sessions.pop(session_id, None)


store = SessionStore()
