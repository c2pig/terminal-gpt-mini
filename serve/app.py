"""OpenAI-compatible chat server for the tiny terminal model.

    MODEL_PATH=ml/outputs/smoke uvicorn serve.app:app --host 127.0.0.1 --port 8000

MODEL_PATH points at a local Hugging Face folder. Otherwise the process loads
HF_MODEL_ID (default c2pig/terminal-shell-mini). HF_TOKEN is used by the Hub
client when the repo is private; do not commit it.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

from chat_format import decode_new_tokens, prompt_token_ids  # noqa: E402

DEFAULT_MODEL_ID = "c2pig/terminal-shell-mini"

_model = None
_tokenizer = None
_source = ""


def model_source() -> str:
    local = os.environ.get("MODEL_PATH", "").strip()
    if local:
        return local
    return os.environ.get("HF_MODEL_ID", "").strip() or DEFAULT_MODEL_ID


def load_model() -> None:
    global _model, _tokenizer, _source
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _source = model_source()
    _tokenizer = AutoTokenizer.from_pretrained(_source)
    _model = AutoModelForCausalLM.from_pretrained(_source)
    _model.eval()
    _model.to("cpu")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_model()
    yield


app = FastAPI(title="terminal-shell-mini", lifespan=lifespan)


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict] = Field(default_factory=list)
    max_tokens: int | None = 128
    temperature: float | None = 0.0
    stream: bool | None = False


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "invalid_request_error"}},
    )


def _coerce_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return str(content or "")


def generate_reply(messages: list[dict], max_tokens: int, temperature: float) -> tuple[str, int, int]:
    assert _model is not None and _tokenizer is not None
    positions = int(_model.config.n_positions)
    max_new = max(1, min(max_tokens, 256, positions - 8))
    budget = max(8, positions - max_new)
    ids = prompt_token_ids(_tokenizer, messages, budget)
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention = torch.ones_like(input_ids)
    end_id = _tokenizer.convert_tokens_to_ids("<|end|>")
    do_sample = temperature is not None and temperature > 0
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention,
        "max_new_tokens": max_new,
        "do_sample": do_sample,
        "pad_token_id": _tokenizer.pad_token_id,
        "eos_token_id": end_id,
    }
    if do_sample:
        kwargs["temperature"] = float(temperature)
    with torch.inference_mode():
        output = _model.generate(**kwargs)
    text = decode_new_tokens(_tokenizer, output[0].tolist(), len(ids))
    completion_tokens = int(output.shape[-1] - len(ids))
    return text, len(ids), max(completion_tokens, 0)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": _source}


@app.get("/v1/models")
def models() -> dict:
    name = os.environ.get("OPENAI_MODEL", "").strip() or "terminal-shell-mini"
    return {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "local"}]}


@app.post("/v1/chat/completions")
def chat_completions(body: ChatRequest):
    if body.stream:
        return _error(400, "stream=true is not supported")
    if not body.messages:
        return _error(400, "messages is required")
    normalized = []
    for message in body.messages:
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            return _error(400, f"unsupported role: {role}")
        normalized.append({"role": role, "content": _coerce_content(message.get("content"))})
    max_tokens = body.max_tokens if body.max_tokens is not None else 128
    temperature = 0.0 if body.temperature is None else float(body.temperature)
    text, prompt_tokens, completion_tokens = generate_reply(normalized, max_tokens, temperature)
    model_name = body.model or os.environ.get("OPENAI_MODEL", "").strip() or "terminal-shell-mini"
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
