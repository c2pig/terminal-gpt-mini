#!/usr/bin/env python3
"""Export all lab/data/*.json used by the 4-wing museum."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    scripts = [
        ROOT / "lab/scripts/export_example.py",
    ]
    for s in scripts:
        if s.is_file():
            subprocess.check_call([sys.executable, str(s)], cwd=ROOT)

    smoke = ROOT / "ml/outputs/smoke"
    if not smoke.is_dir():
        print("Skip inference-trace / tensor-manifest: run ml/train.py --smoke first")
        return 0

    sys.path.insert(0, str(ROOT / "ml"))
    import torch
    from chat_format import decode_new_tokens, ensure_system, prompt_token_ids
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(smoke)
    model = AutoModelForCausalLM.from_pretrained(smoke)
    model.eval()

    messages = ensure_system(
        [
            {
                "role": "user",
                "content": "user@dev:~$ pwd",
            }
        ]
    )
    prompt_ids = prompt_token_ids(tok, messages, 200)
    end_id = tok.convert_tokens_to_ids("<|end|>")
    steps = []
    cur = prompt_ids.copy()
    for i in range(16):
        inp = torch.tensor([cur], dtype=torch.long)
        with torch.inference_mode():
            out = model.generate(
                inp,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=end_id,
            )
        new_id = int(out[0, -1].item())
        cur = cur + [new_id]
        steps.append(
            {
                "step": i + 1,
                "new_id": new_id,
                "piece": tok.decode([new_id]),
                "ids_len": len(cur),
                "decoded_new": decode_new_tokens(tok, cur, len(prompt_ids)),
            }
        )
        if new_id == end_id:
            break

    data_dir = ROOT / "lab/data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "inference-trace.json").write_text(
        json.dumps(
            {
                "prompt_ids": prompt_ids,
                "prompt_len": len(prompt_ids),
                "steps": steps,
                "n_embd": int(model.config.n_embd),
                "vocab_size": len(tok),
            },
            indent=2,
        )
        + "\n"
    )

    manifest = []
    with safe_open(smoke / "model.safetensors", framework="pt") as f:
        for key in sorted(f.keys()):
            t = f.get_tensor(key)
            manifest.append({"name": key, "shape": list(t.shape)})
    (data_dir / "tensor-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported museum data to {data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
