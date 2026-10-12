"""OpenAI-compatible terminal chat.

Interactive: terminal-gpt
One shot:    terminal-gpt "your question"
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from terminal_gpt_mini.prompts import system_prompt_for_model


def _client() -> tuple[OpenAI, str]:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and set it.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    model = os.environ.get("OPENAI_MODEL", "").strip() or "gpt-4o-mini"
    return OpenAI(api_key=api_key, base_url=base_url), model


def complete(client: OpenAI, model: str, messages: list[dict[str, str]]) -> str:
    kwargs: dict = {"model": model, "messages": messages}
    if "terminal-shell" in model:
        kwargs["temperature"] = 0
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    return content or ""


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    client, model = _client()
    messages: list[dict[str, str]] = []
    system = system_prompt_for_model(model)
    if system:
        messages.append({"role": "system", "content": system})

    if args:
        messages.append({"role": "user", "content": " ".join(args)})
        try:
            print(complete(client, model, messages))
        except Exception as exc:  # noqa: BLE001 — surface API errors to the user
            print(f"request failed: {exc}", file=sys.stderr)
            return 1
        return 0

    print("terminal-gpt  (/exit to quit)")
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        stripped = line.strip()
        if stripped in {"/exit", "/quit"}:
            return 0
        if not stripped:
            continue
        messages.append({"role": "user", "content": line})
        try:
            reply = complete(client, model, messages)
        except Exception as exc:  # noqa: BLE001
            print(f"request failed: {exc}", file=sys.stderr)
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": reply})
        print(reply)


if __name__ == "__main__":
    raise SystemExit(main())
