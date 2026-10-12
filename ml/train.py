#!/usr/bin/env python3
"""Train a tiny GPT-2 to imitate Ubuntu shell transcripts.

Examples:
  python ml/train.py --smoke
  python ml/train.py --smoke --max-steps 20 --output-dir ml/outputs/smoke
  python ml/train.py --config ml/config/tiny.yaml --output-dir ml/outputs/full
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

from chat_format import SPECIAL_TOKENS, encode_example, prepare_tokenizer  # noqa: E402

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")


def git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def load_config(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    return path


class EncodedChatDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_seq_len: int):
        raw = load_dataset("json", data_files=str(path), split="train")
        self.rows: list[dict] = []
        skipped = 0
        for record in raw:
            encoded = encode_example(record["messages"], tokenizer, max_seq_len)
            if encoded is None:
                skipped += 1
                continue
            self.rows.append(encoded)
        if not self.rows:
            raise SystemExit(f"No trainable rows in {path} (skipped {skipped}).")
        if skipped:
            print(f"skipped {skipped} rows in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class PadCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        width = max(len(feature["input_ids"]) for feature in features)
        input_ids, labels, attention = [], [], []
        for feature in features:
            pad = width - len(feature["input_ids"])
            input_ids.append(list(feature["input_ids"]) + [self.pad_token_id] * pad)
            labels.append(list(feature["labels"]) + [-100] * pad)
            attention.append(list(feature["attention_mask"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


def unique_parameter_count(model) -> int:
    seen: dict[int, int] = {}
    for param in model.parameters():
        seen[id(param)] = param.numel()
    return sum(seen.values())


def iter_message_text(paths: list[Path]):
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                for message in record["messages"]:
                    content = str(message.get("content") or "")
                    if content:
                        yield content


def build_tokenizer(paths: list[Path], vocab_size: int) -> PreTrainedTokenizerFast:
    """Fit a byte-level BPE on the chat text so the embedding table stays small."""
    backend = Tokenizer(BPE(unk_token="<unk>"))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
    backend.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=["<unk>", "<|endoftext|>", *SPECIAL_TOKENS],
        show_progress=False,
    )
    backend.train_from_iterator(iter_message_text(paths), trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=list(SPECIAL_TOKENS),
    )
    return prepare_tokenizer(tokenizer)


def build_model(cfg: dict, tokenizer) -> GPT2LMHeadModel:
    model_cfg = cfg["model"]
    n_embd = int(model_cfg["n_embd"])
    n_head = int(model_cfg["n_head"])
    if n_embd % n_head != 0:
        raise SystemExit(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=int(model_cfg["n_positions"]),
        n_embd=n_embd,
        n_layer=int(model_cfg["n_layer"]),
        n_head=n_head,
        n_inner=int(model_cfg["n_inner"]),
        resid_pdrop=float(model_cfg.get("resid_pdrop", 0.0)),
        embd_pdrop=float(model_cfg.get("embd_pdrop", 0.0)),
        attn_pdrop=float(model_cfg.get("attn_pdrop", 0.0)),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|end|>"),
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    end_id = tokenizer.convert_tokens_to_ids("<|end|>")
    if model.generation_config is not None:
        model.generation_config.eos_token_id = end_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the tiny terminal-shell model")
    parser.add_argument("--config", default="ml/config/tiny.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Train on the committed sample")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    cfg = load_config(resolve(args.config))
    train_cfg = dict(cfg["train"])
    if args.smoke:
        train_cfg.update(cfg.get("smoke") or {})
    max_steps = args.max_steps if args.max_steps is not None else int(train_cfg["max_steps"])
    max_seq_len = int(train_cfg.get("max_seq_len", cfg["train"]["max_seq_len"]))
    if int(cfg["model"]["n_positions"]) < max_seq_len:
        raise SystemExit("model.n_positions must be >= max_seq_len")

    data_cfg = cfg["data"]
    if args.smoke:
        train_file = resolve(data_cfg["sample_train_file"])
        val_file = resolve(data_cfg["sample_val_file"])
    else:
        train_file = resolve(data_cfg["train_file"])
        val_file = resolve(data_cfg["val_file"])
    for path in (train_file, val_file):
        if not path.is_file():
            raise SystemExit(
                f"Missing {path}. Generate it with ml/data/generate_sessions.py "
                "(see README). Smoke mode uses ml/data/sample."
            )

    output_dir = Path(args.output_dir) if args.output_dir else (
        ROOT / "ml" / "outputs" / ("smoke" if args.smoke else "full")
    )
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = int(cfg.get("tokenizer", {}).get("vocab_size", 800))
    tokenizer = build_tokenizer([train_file, val_file], vocab_size)
    marker = tokenizer.encode("<|system|>", add_special_tokens=False)
    if len(marker) != 1:
        raise SystemExit("Chat marker tokens were not kept intact by the tokenizer.")
    tokenizer.model_max_length = max_seq_len

    train_ds = EncodedChatDataset(train_file, tokenizer, max_seq_len)
    val_ds = EncodedChatDataset(val_file, tokenizer, max_seq_len)
    model = build_model(cfg, tokenizer)
    n_params = unique_parameter_count(model)
    print(f"parameters: {n_params}")
    print(f"train rows: {len(train_ds)}  val rows: {len(val_ds)}  steps: {max_steps}")
    if not 1_000_000 <= n_params <= 10_000_000:
        raise SystemExit(f"parameter count {n_params} is outside the 1M–10M target")

    eval_steps = int(train_cfg.get("eval_steps", 100))
    do_eval = max_steps >= eval_steps
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        max_steps=max_steps,
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 4)),
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        warmup_steps=int(train_cfg.get("warmup_steps", 0)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        eval_strategy="steps" if do_eval else "no",
        eval_steps=eval_steps if do_eval else None,
        save_strategy="no",
        seed=int(train_cfg.get("seed", 42)),
        report_to="none",
        use_cpu=True,
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if do_eval else None,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )
    result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    meta = {
        "parameters": n_params,
        "max_steps": max_steps,
        "smoke": bool(args.smoke),
        "train_loss": result.training_loss,
        "git_sha": os.environ.get("GITHUB_SHA") or git_sha(),
        "train_file": str(train_file.relative_to(ROOT)),
    }
    (output_dir / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"saved {output_dir}")
    print(f"train_loss: {result.training_loss:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
