"""Chat prompt format for c2pig/terminal-shell-mini (Space copy)."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a Linux terminal on Ubuntu. "
    "If the user message is a shell prompt and command, reply with only the command output. "
    "If the user asks a question, answer in one or two short sentences."
)

END_TOKEN = "<|end|>"


def ensure_system(messages: list[dict]) -> list[dict]:
    if messages and messages[0].get("role") == "system":
        return messages
    return [{"role": "system", "content": SYSTEM_PROMPT}, *messages]


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


def prompt_token_ids(tokenizer, messages: list[dict], max_input_len: int) -> list[int]:
    text = render_messages(ensure_system(messages), for_generation=True)
    ids = tokenizer.encode(text, add_special_tokens=False)
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
