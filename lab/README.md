# Knowledge museum (4 wings)

Interactive static SPA for **training and inference** in `terminal-gpt-mini`.

| Wing | Topics |
|------|--------|
| **MAP** | Repo flow: Docker → JSONL → train → Hub → serve / Space / CLI |
| **TRAIN** | `ml/train.py` steps 0–8 (tokenizer, labels, Trainer, …) |
| **ARTIFACTS** | `config.json`, tokenizer, `model.safetensors` |
| **INFER** | Context, IDs vs vectors, generate loop, train vs infer |

## Run

```bash
cd /path/to/terminal-gpt-mini
python3 -m http.server 8080
```

Open **http://localhost:8080/lab/**

## Data files

```bash
python3 lab/scripts/export_museum_data.py
```

Requires `ml/outputs/smoke/` for inference trace and tensor manifest (run `python3 ml/train.py --smoke` first).

| File | Purpose |
|------|---------|
| `data/pwd-example.json` | Real `encode_example` for TRAIN wing |
| `data/repo-map.json` | MAP nodes/edges |
| `data/inference-trace.json` | INFER generate steps from smoke model |
| `data/tensor-manifest.json` | Tensor names/shapes from smoke safetensors |

## UI

- **Gap tour** — guided path through common knowledge gaps
- **Glossary** — weights, safetensors, context, -100, …
