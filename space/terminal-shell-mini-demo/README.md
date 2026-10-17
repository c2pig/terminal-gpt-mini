---
title: terminal-shell-mini demo
emoji: 🖥️
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
python_version: "3.12"
short_description: Fake Ubuntu shell powered by terminal-shell-mini
startup_duration_timeout: 30m
models:
  - c2pig/terminal-shell-mini
---

# terminal-shell-mini demo

Interactive demo for [c2pig/terminal-shell-mini](https://huggingface.co/c2pig/terminal-shell-mini): a tiny GPT-2-style model trained on synthetic Docker bash sessions.

This is **not** a real shell. Outputs are model guesses.

## Usage

Type a command (`pwd`, `whoami`) or a full line (`user@dev:~$ ls`). You can also ask short English questions.

## Source

Built from [terminal-gpt-mini](https://github.com/c2pig/terminal-gpt-mini) (`space/terminal-shell-mini-demo/`).
