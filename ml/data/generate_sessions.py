#!/usr/bin/env python3
"""Capture synthetic bash sessions from ubuntu:22.04 and write chat JSONL.

Smoke tests use the committed sample under ml/data/sample. Regenerate it with:

    python ml/data/generate_sessions.py --sample --out-dir ml/data/sample

Full training data (gitignored) for a local run or GitHub Actions:

    python ml/data/generate_sessions.py --out-dir ml/data/full --variations 3

Requires Docker. The script uses `docker` when the current user can talk to
the daemon, and `sudo docker` otherwise. Set DOCKER to override the binary,
for example DOCKER="sudo docker".
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from terminal_gpt_mini.prompts import SYSTEM_PROMPT  # noqa: E402

IMAGE = "ubuntu:22.04"
CONTAINER = "terminal-gpt-mini-sessions"
HOME = "/home/user"

SETUP = r"""
set -eu
if ! id user >/dev/null 2>&1; then
  useradd -m -s /bin/bash user
fi
rm -rf /home/user/projects /home/user/docs /home/user/bin /home/user/tmpwork /home/user/nested
mkdir -p /home/user/projects/app /home/user/docs /home/user/bin
printf 'print("hi")\n' > /home/user/projects/app/main.py
printf '# app\n' > /home/user/projects/app/README.md
printf 'hello world\nsecond line\n' > /home/user/docs/notes.txt
printf 'alpha\nbeta\nalpha\ngamma\n' > /home/user/docs/words.txt
printf '#!/bin/sh\necho tool\n' > /home/user/bin/tool
chmod 755 /home/user/bin/tool
chown -R user:user /home/user
"""


def docker_base() -> list[str]:
    override = os.environ.get("DOCKER", "").strip()
    if override:
        return override.split()
    try:
        subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return ["docker"]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ["sudo", "docker"]


def run(args: list[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(docker_base() + args, check=check, **kwargs)


def display_cwd(cwd: str) -> str:
    if cwd == HOME:
        return "~"
    prefix = HOME + "/"
    if cwd.startswith(prefix):
        return "~/" + cwd[len(prefix) :]
    return cwd


def prompt_line(cwd: str, command: str) -> str:
    return f"user@dev:{display_cwd(cwd)}$ {command}"


def combine(stdout: str, stderr: str) -> str:
    out = stdout.replace("\r\n", "\n")
    err = stderr.replace("\r\n", "\n")
    if out.endswith("\n"):
        out = out[:-1]
    if err.endswith("\n"):
        err = err[:-1]
    if out and err:
        return out + "\n" + err
    return out or err


def one(cwd: str, command: str, question: str | None = None, answer: str | None = None) -> dict:
    questions = [(question, answer)] if question else []
    return {"commands": [(cwd, command)], "questions": questions}


def multi(commands: list[tuple[str, str]], questions: list[tuple[str, str]] | None = None) -> dict:
    return {"commands": commands, "questions": questions or []}


def curated() -> list[dict]:
    home = HOME
    docs = f"{HOME}/docs"
    app = f"{HOME}/projects/app"
    return [
        one(home, "pwd", "What directory am I in?", "You are in /home/user."),
        one(home, "whoami", "Who is the current user?", "The current user is user."),
        one(home, "echo hello"),
        one(home, "echo $HOME"),
        one(home, "echo $((2+2))"),
        one(home, "printf 'a\\nb\\n'"),
        one(home, "uname -s"),
        one(home, "uname -m"),
        one(home, "which bash"),
        one(home, "command -v ls"),
        one(home, "ls /no/such", "Did that command succeed?", "No. ls reported that the path does not exist."),
        one(home, "cat /home/user/missing.txt"),
        one(home, "true; echo $?"),
        one(home, "false; echo $?"),
        one(home, "seq 1 5"),
        one(docs, "wc -l notes.txt"),
        one(docs, "head -n 2 notes.txt"),
        one(docs, "tail -n 1 notes.txt"),
        one(docs, "grep alpha words.txt"),
        one(docs, "sort words.txt"),
        one(docs, "cat notes.txt", "What is in notes.txt?", "Two lines: hello world, then second line."),
        one(home, "ls", "Which entries are in this directory?", "bin, docs, and projects."),
        one(home, "ls projects"),
        one(home, "ls projects/app"),
        one(app, "cat main.py"),
        one(home, "basename /home/user/docs/notes.txt"),
        one(home, "dirname /home/user/docs/notes.txt"),
        one(home, "date -u -d @0 +%Y-%m-%d"),
        multi(
            [(home, "mkdir -p tmpwork"), (home, "ls tmpwork")],
            [("Is tmpwork empty?", "Yes. ls printed nothing.")],
        ),
        multi(
            [(home, "mkdir -p tmpwork"), (home, "touch tmpwork/a.txt"), (home, "ls tmpwork")],
        ),
        one(home, "echo hello world"),
        one(home, "printf 'a\\nb\\na\\n' | wc -l"),
        one(docs, "sort words.txt | uniq"),
        one(docs, "cut -d' ' -f1 notes.txt"),
        one(home, "printf 'hi\\n' | tr a-z A-Z"),
        one(home, "printf 'abc\\n' | rev"),
        one(docs, "wc -w notes.txt"),
        one(docs, "wc -c notes.txt"),
        one(home, "echo -n hello"),
        one(docs, "pwd"),
        one(home, "cd docs && pwd", "Where did cd go?", "It changed into /home/user/docs."),
        one(docs, "ls"),
        one(docs, "grep world notes.txt"),
        one(docs, "head -n 1 notes.txt"),
        one(docs, "tail -n 2 notes.txt"),
        one(home, "echo $((10*3))"),
        one(home, "echo $((7-2))"),
        one(home, "seq 1 3"),
        one(home, "printf '%s\\n' one two"),
        one(home, "which ls"),
        one(home, "expr 3 + 4", "What is 3 + 4 in this shell?", "expr printed 7."),
        one(home, "id -un"),
        one(home, "echo user"),
        one(home, "ls bin"),
        one(home, "/home/user/bin/tool"),
        multi([(home, "mkdir -p nested/leaf"), (home, "ls nested")]),
        multi(
            [
                (home, "mkdir -p tmpwork"),
                (home, "printf 'one\\n' > tmpwork/one.txt"),
                (home, "cat tmpwork/one.txt"),
            ]
        ),
        multi([(docs, "cp notes.txt notes.bak"), (docs, "ls")]),
        one(docs, "diff notes.txt words.txt"),
        one(docs, "cat -n notes.txt"),
        one(docs, "wc -l words.txt"),
        one(home, "printf 'x\\ny\\nx\\n' | sort | uniq -c"),
        one(home, "yes | head -n 3"),
        one(home, "printf 'a\\nb\\n' | wc -l"),
        one(home, "printf 'aaab\\n' | tr -s a"),
        one(home, "printf 'a,b,c\\n' | cut -d, -f2"),
        one(docs, "sort -r words.txt"),
        multi(
            [(home, "pwd"), (home, "ls"), (home, "ls projects")],
            [("Name the home directory and its folders.", "The directory is /home/user, with bin, docs, and projects.")],
        ),
        multi(
            [(home, "echo shell"), (home, "echo mini")],
            [("What did the two echo commands print?", "They printed shell, then mini.")],
        ),
        one(home, "uname -o"),
        one(home, "arch"),
        one(home, "readlink -f /home/user/docs/notes.txt"),
        one(home, "realpath /home/user/docs/notes.txt"),
        one(docs, "du -b notes.txt"),
        one(docs, "ls -l notes.txt"),
        one(home, "echo *"),
        one(home, "printf '%s\\n' projects/*"),
        one(home, "test -d projects && echo yes"),
        one(home, "test -f missing && echo yes || echo no"),
        one(home, "basename $(pwd)"),
        one(docs, "wc -l < notes.txt"),
        one(docs, "grep -c alpha words.txt"),
        one(docs, "grep -v alpha words.txt"),
        one(docs, "sort -u words.txt"),
        one(app, "wc -l main.py"),
        one(app, "head -n 1 README.md"),
        one(home, "echo $((1+2+3))"),
        one(home, "printf '%d\\n' 4"),
        one(home, "ls /usr/bin/bash"),
        one(home, "type type"),
        one(docs, "grep -n second notes.txt"),
        one(home, "seq 3 5"),
    ]


def variation_sessions(rounds: int) -> list[dict]:
    words = ["hello", "world", "alpha", "beta", "gamma", "delta", "shell", "ubuntu", "bash", "mini"]
    sessions: list[dict] = []
    for round_index in range(rounds):
        suffix = "" if round_index == 0 else f"-{round_index + 1}"
        for word in words:
            text = f"{word}{suffix}"
            question = []
            if round_index == 0 and word in {"hello", "world"}:
                question = [(f"What did that print?", text)]
            sessions.append(
                {
                    "commands": [(HOME, f"echo {text}")],
                    "questions": question,
                }
            )
        for number in (2, 4, 6):
            sessions.append(
                {
                    "commands": [(HOME, f"seq 1 {number + round_index}")],
                    "questions": [],
                }
            )
            sessions.append(
                {
                    "commands": [(HOME, f"echo $(({number}+{round_index}))")],
                    "questions": [],
                }
            )
    return sessions


def assign(sessions: list[dict]) -> list[dict]:
    assigned = []
    for index, session in enumerate(sessions):
        item = dict(session)
        item["id"] = item.get("id") or f"s{index:03d}"
        item["split"] = "test" if index % 10 == 0 else "val" if index % 10 == 1 else "train"
        assigned.append(item)
    return assigned


class Collector:
    def __init__(self, image: str):
        self.image = image
        self.base = docker_base()

    def start(self) -> None:
        run(["rm", "-f", CONTAINER], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        run(["pull", self.image], check=True)
        run(["run", "-d", "--name", CONTAINER, self.image, "sleep", "infinity"], check=True)

    def stop(self) -> None:
        run(["rm", "-f", CONTAINER], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def reset(self) -> None:
        proc = run(["exec", CONTAINER, "bash", "-lc", SETUP], check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"fixture setup failed:\n{proc.stderr}")

    def capture(self, cwd: str, command: str) -> str:
        proc = run(
            ["exec", "-u", "user", "-w", cwd, CONTAINER, "bash", "-lc", command],
            check=False,
            capture_output=True,
            text=True,
        )
        return combine(proc.stdout or "", proc.stderr or "")


def row_from(session: dict, outputs: list[str]) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for (cwd, command), output in zip(session["commands"], outputs, strict=True):
        messages.append({"role": "user", "content": prompt_line(cwd, command)})
        messages.append({"role": "assistant", "content": output})
    for question, answer in session["questions"]:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
    return {"messages": messages}


def prompts_from(session_id: str, row: dict) -> list[dict]:
    messages = row["messages"]
    prompts = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant" or index == 0:
            continue
        user = messages[index - 1]
        if user["role"] != "user":
            continue
        prompts.append(
            {
                "id": f"{session_id}-{len(prompts)}",
                "messages": messages[:index],
                "reference": message["content"],
            }
        )
    return prompts


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def push_dataset(out_dir: Path, repo_id: str) -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is required for --push-dataset")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(out_dir),
        repo_type="dataset",
        commit_message="Update terminal shell sessions",
    )
    print(f"pushed dataset to https://huggingface.co/datasets/{repo_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate terminal chat JSONL from Docker")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample", action="store_true", help="Curated sample only (commit this)")
    parser.add_argument("--variations", type=int, default=3, help="Extra echo/seq rounds for a full set")
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--push-dataset", default=None, help="Optional Hub dataset repo id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sessions = curated()
    if not args.sample:
        sessions.extend(variation_sessions(args.variations))
    sessions = assign(sessions)
    collector = Collector(args.image)
    buckets = {"train": [], "val": [], "test": []}
    eval_prompts: list[dict] = []
    collector.start()
    try:
        for session in sessions:
            collector.reset()
            outputs = [collector.capture(cwd, command) for cwd, command in session["commands"]]
            row = row_from(session, outputs)
            buckets[session["split"]].append(row)
            if session["split"] == "test":
                eval_prompts.extend(prompts_from(session["id"], row))
    finally:
        collector.stop()

    out_dir = args.out_dir
    write_jsonl(out_dir / "train.jsonl", buckets["train"])
    write_jsonl(out_dir / "val.jsonl", buckets["val"])
    write_jsonl(out_dir / "test.jsonl", buckets["test"])
    prompt_path = out_dir / "eval_prompts.json"
    if args.sample:
        prompt_path.write_text(json.dumps(eval_prompts, indent=2) + "\n")
    counts = {name: len(rows) for name, rows in buckets.items()}
    total = sum(counts.values())
    print(f"wrote {total} sessions to {out_dir} ({counts})")
    if args.push_dataset:
        push_dataset(out_dir, args.push_dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
