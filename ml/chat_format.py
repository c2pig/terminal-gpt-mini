"""Chat markup shared by training, eval, and the local OpenAI-compatible server.

The string layout must stay identical to the ids produced by encode_example.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from terminal_gpt_mini.prompts import SYSTEM_PROMPT  # noqa: E402

SPECIAL_TOKENS = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
END_TOKEN = "<|end|>"


def prepare_tokenizer(tokenizer):
    vocab = tokenizer.get_vocab()
    missing = [token for token in SPECIAL_TOKENS if token not in vocab]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _ids(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def render_messages(messages: list[dict], *, for_generation: bool) -> str:
    chunks: list[str] = []
    for message in messages:
        role = message["role"]
        content = str(message.get("content") or "").rstrip("\n")
        chunks.append(f"<|{role}|>\n")
        if content:
            chunks.append(f"{content}\n")
        if role == "assistant":
            chunks.append(f"{END_TOKEN}\n")
    if for_generation and (not messages or messages[-1]["role"] != "assistant"):
        chunks.append("<|assistant|>\n")
    return "".join(chunks)


def encode_example(messages: list[dict], tokenizer, max_seq_len: int) -> dict | None:
    """Tokenize one conversation. Loss is applied only to assistant text and the end marker."""
    input_ids: list[int] = []
    labels: list[int] = []
    for message in messages:
        role = message["role"]
        content = str(message.get("content") or "").rstrip("\n")
        header = _ids(tokenizer, f"<|{role}|>\n")
        body = _ids(tokenizer, f"{content}\n") if content else []
        input_ids.extend(header)
        input_ids.extend(body)
        if role == "assistant":
            labels.extend([-100] * len(header))
            labels.extend(body)
            end = _ids(tokenizer, f"{END_TOKEN}\n")
            input_ids.extend(end)
            labels.extend(end)
        else:
            labels.extend([-100] * (len(header) + len(body)))
    if max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]
    if not any(label != -100 for label in labels):
        return None
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def ensure_system(messages: list[dict]) -> list[dict]:
    if messages and messages[0].get("role") == "system":
        return messages
    return [{"role": "system", "content": SYSTEM_PROMPT}, *messages]


def prompt_token_ids(tokenizer, messages: list[dict], max_input_len: int) -> list[int]:
    text = render_messages(ensure_system(messages), for_generation=True)
    ids = _ids(tokenizer, text)
    if max_input_len > 0 and len(ids) > max_input_len:
        ids = ids[-max_input_len:]
    return ids


def decode_new_tokens(tokenizer, generated_ids, prompt_len: int) -> str:
    new_ids = list(generated_ids)[prompt_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=False)
    for stop in (END_TOKEN, "<|user|>", "<|system|>", "<|assistant|>"):
        if stop in text:
            text = text.split(stop, 1)[0]
    return text.rstrip("\n")
