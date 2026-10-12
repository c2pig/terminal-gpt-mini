#!/usr/bin/env python3
"""Evaluate a terminal-shell checkpoint: held-out loss and a few generations.

  python ml/eval.py --model-dir ml/outputs/smoke
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

from chat_format import decode_new_tokens, prompt_token_ids  # noqa: E402
from train import EncodedChatDataset, PadCollator, load_config, resolve  # noqa: E402


def pick_test_file(cfg: dict, model_dir: Path) -> Path:
    data = cfg["data"]
    if model_dir.name == "smoke":
        return resolve(data["sample_test_file"])
    full = resolve(data["test_file"])
    if full.is_file():
        return full
    return resolve(data["sample_test_file"])


def load_prompts(path: Path, limit: int) -> list[dict]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or not payload:
        raise SystemExit(f"{path} must be a non-empty JSON list")
    return payload[:limit]


def mean_loss(model, dataset, pad_token_id: int) -> float:
    loader = DataLoader(dataset, batch_size=4, collate_fn=PadCollator(pad_token_id))
    total = 0.0
    seen = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            batch_size = batch["input_ids"].size(0)
            total += float(outputs.loss) * batch_size
            seen += batch_size
    if seen == 0:
        raise SystemExit("evaluation dataset is empty")
    return total / seen


def generate_one(model, tokenizer, messages: list[dict], max_new_tokens: int) -> str:
    positions = int(model.config.n_positions)
    budget = max(8, positions - max_new_tokens)
    ids = prompt_token_ids(tokenizer, messages, budget)
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention = torch.ones_like(input_ids)
    end_id = tokenizer.convert_tokens_to_ids("<|end|>")
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=end_id,
        )
    return decode_new_tokens(tokenizer, output[0].tolist(), len(ids))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a terminal-shell checkpoint")
    parser.add_argument("--model-dir", default="ml/outputs/smoke")
    parser.add_argument("--config", default="ml/config/tiny.yaml")
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--generations", default=None)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = resolve(args.model_dir)
    if not (model_dir / "config.json").is_file():
        raise SystemExit(f"No config.json in {model_dir}. Train a checkpoint first.")
    cfg = load_config(resolve(args.config))
    test_file = resolve(args.test_file) if args.test_file else pick_test_file(cfg, model_dir)
    prompts_file = resolve(args.prompts or cfg["data"]["eval_prompts_file"])
    if not test_file.is_file():
        raise SystemExit(f"Missing test file {test_file}")
    if not prompts_file.is_file():
        raise SystemExit(f"Missing prompts file {prompts_file}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir))
    model.eval()
    max_seq_len = int(cfg["train"]["max_seq_len"])
    dataset = EncodedChatDataset(test_file, tokenizer, max_seq_len)
    loss = mean_loss(model, dataset, tokenizer.pad_token_id)
    if math.isnan(loss) or math.isinf(loss):
        raise SystemExit(f"evaluation loss is not finite: {loss}")
    perplexity = math.exp(loss) if loss < 20 else None

    generations = []
    lines = []
    for item in load_prompts(prompts_file, args.limit):
        messages = item["messages"]
        # Drop a trailing assistant turn so we score the model's completion.
        if messages and messages[-1]["role"] == "assistant":
            messages = messages[:-1]
        text = generate_one(model, tokenizer, messages, args.max_new_tokens)
        user = next(message["content"] for message in reversed(messages) if message["role"] == "user")
        record = {
            "id": item.get("id"),
            "user": user,
            "reference": item.get("reference"),
            "generated": text,
        }
        generations.append(record)
        lines.append(f"### {record['id']}\nUSER: {user}\nREF: {record['reference']}\nGEN: {text}\n")

    payload = {
        "model_dir": str(model_dir),
        "test_file": str(test_file),
        "num_eval_examples": len(dataset),
        "loss": loss,
        "perplexity": perplexity,
        "generations": generations,
    }
    output = Path(args.output) if args.output else model_dir / "eval_results.json"
    generations_path = Path(args.generations) if args.generations else model_dir / "sample_generations.txt"
    if not output.is_absolute():
        output = ROOT / output
    if not generations_path.is_absolute():
        generations_path = ROOT / generations_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    generations_path.write_text("\n".join(lines))
    print(f"loss: {loss:.4f}")
    if perplexity is not None:
        print(f"perplexity: {perplexity:.3f}")
    print(f"wrote {output}")
    print(f"wrote {generations_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
