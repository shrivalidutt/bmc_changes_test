"""
Local TinyLlama inference (no Ollama server).

Set TINYLLAMA_MODEL to override the Hugging Face model id
(default: TinyLlama/TinyLlama-1.1B-Chat-v1.0).
"""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

# Limit CPU threads to prevent thread-switching CPU thrashing
torch.set_num_threads(4)

from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

MODEL_ID = os.getenv("TINYLLAMA_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

_model = None
_tokenizer = None
_device = None


@dataclass
class LLMResult:
    content: str


def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _hf_kwargs():
    return {"token": HF_TOKEN} if HF_TOKEN else {}


def _ensure_model_cached() -> str:
    """Download with a visible tqdm progress bar; resume if interrupted."""
    from huggingface_hub import snapshot_download

    print(f"\n📥 Model cache check: {MODEL_ID}", flush=True)
    print("   (~2.2 GB — if downloading, a progress bar appears below)\n", flush=True)
    return snapshot_download(
        repo_id=MODEL_ID,
        token=HF_TOKEN,
    )


def _load_model():
    global _model, _tokenizer, _device
    if _model is not None:
        return _model, _tokenizer, _device

    _device = _pick_device()
    dtype = torch.float16 if _device != "cpu" else torch.bfloat16
    hf = _hf_kwargs()

    t0 = time.time()
    local_dir = _ensure_model_cached()
    print(f"⏳ Loading weights into memory on {_device}...", flush=True)

    _tokenizer = AutoTokenizer.from_pretrained(local_dir, **hf)
    _model = AutoModelForCausalLM.from_pretrained(local_dir, dtype=dtype, **hf)
    _model.to(_device)
    _model.eval()
    if hasattr(_model, "generation_config") and _model.generation_config is not None:
        _model.generation_config.max_length = None

    print(f"   Model ready in {time.time() - t0:.1f}s\n", flush=True)
    return _model, _tokenizer, _device


def warmup_llm():
    """Load weights before the first user message so chat feels responsive."""
    model, tokenizer, device = _load_model()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    print("   Warmup complete — you can type your request.\n", flush=True)


def _format_chat_prompt(user_text: str) -> str:
    _, tokenizer, _ = _load_model()
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _generate_text(prompt: str, max_new_tokens: int, temperature: float) -> str:
    model, tokenizer, device = _load_model()
    chat_prompt = _format_chat_prompt(prompt)
    inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs["do_sample"] = False

    t0 = time.time()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    elapsed = time.time() - t0
    if os.getenv("LLM_DEBUG"):
        print(f"   [llm {elapsed:.1f}s, max_new_tokens={max_new_tokens}]", flush=True)
    return text


class TinyLlamaLLM:
    """LangChain-compatible surface: invoke(prompt) -> object with .content."""

    def __init__(self, max_new_tokens: int = 256, temperature: float = 0.0):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def invoke(self, prompt: str) -> LLMResult:
        return LLMResult(
            content=_generate_text(prompt, self.max_new_tokens, self.temperature)
        )


def create_llm(max_new_tokens: int = 256) -> TinyLlamaLLM:
    return TinyLlamaLLM(max_new_tokens=max_new_tokens, temperature=0.0)
