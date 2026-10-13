"""Gradio demo for https://huggingface.co/c2pig/terminal-shell-mini"""

from __future__ import annotations

import os

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
def _zerogpu_ready() -> str:
    """Satisfies ZeroGPU hosting; inference runs on CPU below."""
    return "ok"


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


def run_command(
    user_input: str,
    history: list[tuple[str, str]],
) -> tuple[str, list[tuple[str, str]]]:
    """Run one fake-shell turn: user line in, model stdout-style text out."""
    _zerogpu_ready()
    user_line = _to_user_message(user_input)
    if not user_line:
        return "", history

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
    reply = decode_new_tokens(_tokenizer, output[0].tolist(), len(prompt_len_ids))
    history = history + [(user_input.strip(), reply)]
    return "", history


def _pairs_from_messages(history: list | None) -> list[tuple[str, str]]:
    if not history:
        return []
    pairs: list[tuple[str, str]] = []
    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = str(item.get("content") or "")
            if role == "user":
                pairs.append((content, ""))
            elif role == "assistant" and pairs:
                u, _ = pairs[-1]
                pairs[-1] = (u, content)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            pairs.append((str(item[0]), str(item[1])))
    return [(u, a) for u, a in pairs if a or u]


def _messages_from_pairs(pairs: list[tuple[str, str]]) -> list[dict]:
    out: list[dict] = []
    for user, assistant in pairs:
        out.append({"role": "user", "content": user})
        out.append({"role": "assistant", "content": assistant})
    return out


def respond(message: str, history: list | None) -> tuple[str, list]:
    """Append one user turn and the model reply to the chat history."""
    pairs = _pairs_from_messages(history)
    _, new_pairs = run_command(message, pairs)
    return "", _messages_from_pairs(new_pairs)


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
    clear.click(lambda: (None, ""), outputs=[chat, box])

if __name__ == "__main__":
    demo.launch(mcp_server=True)
