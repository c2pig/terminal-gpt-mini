#!/usr/bin/env python3
"""Export lab/data/pwd-example.json from real ml/chat_format + smoke tokenizer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from chat_format import encode_example, ensure_system, render_messages  # noqa: E402
from train import build_tokenizer, load_config, resolve  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def main() -> int:
    train_file = ROOT / "ml/data/sample/train.jsonl"
    row = None
    for line in train_file.read_text().strip().split("\n"):
        record = json.loads(line)
        blob = json.dumps(record)
        if "pwd" in blob and "assistant" in blob:
            row = record
            break
    if row is None:
        row = json.loads(train_file.read_text().strip().split("\n")[0])

    smoke_dir = ROOT / "ml/outputs/smoke"
    if smoke_dir.is_dir():
        tok = AutoTokenizer.from_pretrained(smoke_dir)
    else:
        cfg = load_config(ROOT / "ml/config/tiny.yaml")
        train_f = resolve(cfg["data"]["sample_train_file"])
        val_f = resolve(cfg["data"]["sample_val_file"])
        vocab = int(cfg.get("tokenizer", {}).get("vocab_size", 800))
        tok = build_tokenizer([train_f, val_f], vocab)

    max_seq = 256
    enc = encode_example(row["messages"], tok, max_seq)
    if enc is None:
        raise SystemExit("encode_example returned None")

    rendered = render_messages(ensure_system(row["messages"]), for_generation=False)

    def token_spans(ids: list[int]) -> list[dict]:
        return [{"id": tid, "piece": tok.decode([tid])} for tid in ids]

    cfg = load_config(ROOT / "ml/config/tiny.yaml")
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    payload = {
        "messages": row["messages"],
        "rendered": rendered,
        "encoded": enc,
        "token_spans": token_spans(enc["input_ids"]),
        "vocab_size": len(tok),
        "yaml_snippet": {
            "n_layer": int(model_cfg["n_layer"]),
            "n_embd": int(model_cfg["n_embd"]),
            "n_head": int(model_cfg["n_head"]),
            "n_inner": int(model_cfg["n_inner"]),
            "n_positions": int(model_cfg["n_positions"]),
            "max_seq_len": int(train_cfg["max_seq_len"]),
            "vocab_size": int(cfg.get("tokenizer", {}).get("vocab_size", 800)),
            "max_steps_smoke": int(cfg.get("smoke", {}).get("max_steps", 500)),
            "batch_size": int(train_cfg["per_device_train_batch_size"]),
        },
    }
    out = ROOT / "lab/data/pwd-example.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {out} ({len(enc['input_ids'])} tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
