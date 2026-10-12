# AGENTS.md

## Cursor Cloud

Cloud agents smoke-test the tiny shell model. Full training stays on GitHub Actions (`ML train`), which has the CPU time and the `HF_TOKEN` secret.

The committed `.cursor/environment.json` install does three things:

1. `python3 -m pip install -e .` so `terminal-gpt` is on `PATH`.
2. `python3 -m pip install -r ml/requirements-ml.txt` (CPU torch from the PyTorch wheel index).
3. `python3 ml/train.py --smoke`, which trains on `ml/data/sample` for the step count in `ml/config/tiny.yaml` (`smoke.max_steps`, 200).

### Regenerate data

Session capture runs curated commands inside Docker image `ubuntu:22.04`. Cloud install does not start Docker and does not regenerate data; smoke training uses the committed sample so a Cloud build does not need Docker-in-Docker.

On a machine with Docker, or in the `ML train` workflow (GitHub-hosted runners already have Docker):

```bash
python ml/data/generate_sessions.py --sample --out-dir ml/data/sample
python ml/data/generate_sessions.py --out-dir ml/data/full --variations 3
```

`--sample` rewrites the small JSONL checked into git. The full set is gitignored under `ml/data/full/`.

### Smoke train and eval

```bash
python ml/train.py --smoke
python ml/eval.py --model-dir ml/outputs/smoke
```

A shorter run for a pull-request check is `python ml/train.py --smoke --max-steps 10`. Do not run the full `max_steps` budget from `tiny.yaml` in Cursor Cloud.
