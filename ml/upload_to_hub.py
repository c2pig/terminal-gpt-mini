#!/usr/bin/env python3
"""Upload a trained folder to the Hugging Face Hub.

Requires HF_TOKEN with write access. The token is read from the environment
and is never written to the repo.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).resolve().parent / "MODEL_CARD.md.template"


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload terminal-shell-mini to the Hub")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--repo-id", default="c2pig/terminal-shell-mini")
    parser.add_argument("--sha", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("HF_TOKEN is not set. Add it as a GitHub Actions secret or export it locally.", file=sys.stderr)
        return 1
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    if not (model_dir / "config.json").is_file():
        print(f"No model in {model_dir}", file=sys.stderr)
        return 1

    sha = args.sha or os.environ.get("GITHUB_SHA") or git_sha() or "unknown"
    params = "unknown"
    meta_path = model_dir / "train_meta.json"
    if meta_path.is_file():
        params = str(json.loads(meta_path.read_text()).get("parameters", "unknown"))
    card = TEMPLATE.read_text().replace("__GIT_SHA__", sha).replace("__PARAMS__", params)
    (model_dir / "README.md").write_text(card)

    from huggingface_hub import HfApi

    print(f"git SHA: {sha}")
    print(f"uploading {model_dir} to {args.repo_id}")
    api = HfApi(token=token)
    try:
        api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
    except Exception as exc:  # noqa: BLE001 — upload can still succeed if the repo exists
        print(f"create_repo skipped: {exc}")
    info = api.upload_folder(
        repo_id=args.repo_id,
        folder_path=str(model_dir),
        repo_type="model",
        commit_message=f"Upload terminal-shell-mini @ {sha}",
    )
    print(f"uploaded commit: {info.commit_url if hasattr(info, 'commit_url') else info}")
    print(f"https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
