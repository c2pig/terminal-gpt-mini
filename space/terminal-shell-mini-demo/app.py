"""Gradio demo for https://huggingface.co/c2pig/terminal-shell-mini"""

from __future__ import annotations

import os
import traceback

import spaces
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from chat_format import decode_new_tokens, prompt_token_ids

MODEL_ID = os.environ.get("HF_MODEL_ID", "c2pig/terminal-shell-mini")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "96"))

print(f"Loading {MODEL_ID} on CPU…")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
_model.eval()
print("Model ready.")


@spaces.GPU(duration=5)
def _zerogpu_entrypoint():
    """Required for ZeroGPU Spaces; do not call from Gradio handlers (inference is CPU)."""
    return "ok"


def _message_text(content) -> str:
    """Normalize Gradio 6 chat content (str or list of text blocks)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif "text" in block:
                    parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _to_user_message(text: str) -> str:
    """Turn typed input into a training-style user line."""
    line = (text or "").strip()
    if not line:
        return ""
    if line.startswith("user@dev:"):
        return line
    if line.startswith("$"):
        line = line[1:].strip()
    return f"user@dev:~$ {line}"


def _messages_from_history(history: list[tuple[str, str]], user_line: str) -> list[dict]:
    messages: list[dict] = []
    for user, assistant in history:
        if user:
            messages.append({"role": "user", "content": _to_user_message(user)})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": _to_user_message(user_line)})
    return messages


def _infer(user_input: str, history: list[tuple[str, str]]) -> str:
    messages = _messages_from_history(history, user_input)
    positions = int(_model.config.n_positions)
    max_new = max(8, min(MAX_NEW_TOKENS, positions - 8))
    budget = max(8, positions - max_new)
    prompt_len_ids = prompt_token_ids(_tokenizer, messages, budget)
    input_ids = torch.tensor([prompt_len_ids], dtype=torch.long)
    end_id = _tokenizer.convert_tokens_to_ids("<|end|>")

    with torch.inference_mode():
        output = _model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=_tokenizer.pad_token_id,
            eos_token_id=end_id,
        )
    return decode_new_tokens(_tokenizer, output[0].tolist(), len(prompt_len_ids))


def _pairs_from_messages(history: list | None) -> list[tuple[str, str]]:
    if not history:
        return []
    pairs: list[tuple[str, str]] = []
    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = _message_text(item.get("content"))
            if role == "user":
                pairs.append((content, ""))
            elif role == "assistant" and pairs:
                user, _ = pairs[-1]
                pairs[-1] = (user, content)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            pairs.append((str(item[0]), str(item[1])))
    return [(u, a) for u, a in pairs if u]


def _messages_from_pairs(pairs: list[tuple[str, str]]) -> list[dict]:
    out: list[dict] = []
    for user, assistant in pairs:
        out.append({"role": "user", "content": user, "metadata": None})
        if assistant:
            out.append({"role": "assistant", "content": assistant, "metadata": None})
    return out


def respond(message: str, history: list | None) -> tuple[str, list]:
    """Append one user turn and the model reply to the chat history."""
    try:
        text = (message or "").strip()
        if not text:
            return "", history or []
        pairs = _pairs_from_messages(history)
        reply = _infer(text, pairs)
        pairs = pairs + [(text, reply)]
        return "", _messages_from_pairs(pairs)
    except Exception:
        err = traceback.format_exc()
        print(err)
        pairs = _pairs_from_messages(history)
        pairs = pairs + [(message.strip(), f"(error) {err.splitlines()[-1]}")]
        return "", _messages_from_pairs(pairs)


DESCRIPTION = """
**terminal-shell-mini** (~3.4M params) imitates Ubuntu bash output. It is not a real shell.

- Type a command (`pwd`, `whoami`) or a full prompt line (`user@dev:~$ ls`).
- Ask a short English question about the session.

[Model card](https://huggingface.co/c2pig/terminal-shell-mini) · [Source](https://github.com/c2pig/terminal-gpt-mini)
"""

EXAMPLES = [
    ["pwd"],
    ["user@dev:~$ echo $HOME"],
    ["uname -s"],
    ["ls /no/such"],
    ["What directory am I in?"],
]

with gr.Blocks(title="terminal-shell-mini") as demo:
    gr.Markdown("# 🖥️ terminal-shell-mini demo")
    gr.Markdown(DESCRIPTION)
    chat = gr.Chatbot(height=420, label="Session")
    with gr.Row():
        box = gr.Textbox(
            label="Command or question",
            placeholder="pwd  or  user@dev:~$ whoami",
            scale=4,
        )
        send = gr.Button("Run", variant="primary", scale=1)
    gr.Examples(examples=EXAMPLES, inputs=box, label="Try these")
    clear = gr.Button("Clear session")

    box.submit(respond, [box, chat], [box, chat])
    send.click(respond, [box, chat], [box, chat])
    clear.click(lambda: ([], ""), outputs=[chat, box])

if __name__ == "__main__":
    demo.launch()
