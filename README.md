# terminal-gpt-mini

Minimal terminal chat for [OpenAI-compatible](https://platform.openai.com/docs/api-reference/chat) APIs, plus a tiny local model that imitates Ubuntu bash output.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env   # set OPENAI_API_KEY
```

## Usage

```bash
terminal-gpt                    # interactive chat (/exit to quit)
terminal-gpt "your question"    # single reply
```

Optional: `OPENAI_BASE_URL`, `OPENAI_MODEL` (default `gpt-4o-mini`) — see `.env.example`.

When `OPENAI_MODEL` contains `terminal-shell`, the client sends the shell system prompt and `temperature=0`. Any other model name keeps the original client behavior (no system message).

Requires Python 3.11+.

## Interactive training lab (4-wing museum)

Browser SPA: **MAP** (repo flow), **TRAIN** (`train.py`), **ARTIFACTS** (safetensors bundle), **INFER** (context & generate):

```bash
python3 -m http.server 8080
# open http://localhost:8080/lab/
```

See `lab/README.md`. Regenerate data: `python3 lab/scripts/export_museum_data.py`.

## Live demo (Hugging Face Space)

Try the model in the browser (no local install):

**https://huggingface.co/spaces/c2pig/terminal-shell-mini-demo**

Source for the Space lives in `space/terminal-shell-mini-demo/`. To redeploy after edits:

```bash
hf upload c2pig/terminal-shell-mini-demo space/terminal-shell-mini-demo --repo-type space \
  --exclude "**/__pycache__/**"
```

## End-to-end

The local model is a from-scratch GPT-2 of about 3.4M parameters with a byte-level BPE tokenizer fit on the training text. Weights are not stored in git. GitHub Actions trains on CPU and uploads them to Hugging Face.

### 1. Prerequisites

- Docker, for regenerating sessions (`ubuntu:22.04`).
- A Hugging Face user (this repo uses `c2pig`).
- A token with **write** access to the model repo. Fine-grained tokens need "Write" on the contents of `c2pig/terminal-shell-mini`. A classic token needs the Write role. If you also push the dataset, the token needs write access to that dataset repo.
- GitHub Actions secret **`HF_TOKEN`** (the name is exact). In the GitHub repo: Settings → Secrets and variables → Actions → New repository secret. Paste the token as the value. Do not commit it.

Create the empty model repo yourself before the first upload:

1. Open [huggingface.co/new](https://huggingface.co/new) and create model `terminal-shell-mini` under `c2pig` (`c2pig/terminal-shell-mini`).
2. Optional: create dataset `terminal-shell-sessions` (`datasets/c2pig/terminal-shell-sessions`).

### 2. Generate data

The committed sample under `ml/data/sample/` (train, val, test, and `eval_prompts.json`) is enough for smoke tests. Refresh it from Docker with:

```bash
python ml/data/generate_sessions.py --sample --out-dir ml/data/sample
```

Full data is larger and gitignored. Generate it locally, or let the `ML train` workflow do it (that job runs this command on `ubuntu-latest`, which already has Docker):

```bash
python ml/data/generate_sessions.py --out-dir ml/data/full --variations 3
```

Optional dataset upload (needs `HF_TOKEN`):

```bash
python ml/data/generate_sessions.py --out-dir ml/data/full --variations 3 \
  --push-dataset c2pig/terminal-shell-sessions
```

Each JSONL row is a chat: a system message, then `user@dev:~$ <command>` turns whose assistant text is only the captured stdout/stderr, plus a few plain-English follow-ups.

### 3. Smoke train locally

```bash
pip install -e .
pip install -r ml/requirements-ml.txt
python ml/train.py --smoke
python ml/eval.py --model-dir ml/outputs/smoke
```

`--smoke` trains on `ml/data/sample` for `smoke.max_steps` in `ml/config/tiny.yaml` (500). Override with `--max-steps`. The checkpoint is `ml/outputs/smoke/` (`config.json`, `model.safetensors`, tokenizer). `eval.py` writes `eval_results.json` and `sample_generations.txt` next to it.

### 4. Run the GitHub Actions train

1. Confirm the secret `HF_TOKEN` exists (step 1) and the empty Hub model repo exists.
2. GitHub → Actions → **ML train** → Run workflow.
3. Leave **max_steps** empty to use `train.max_steps` in `ml/config/tiny.yaml` (1500). Set it only to shorten a dispatch.
4. The job generates Docker sessions, trains on CPU, runs `eval.py`, uploads `eval_results.json` and `sample_generations.txt` as artifacts, and uploads the folder to `c2pig/terminal-shell-mini`.
5. The upload step log prints `git SHA: <commit>`. The same SHA is written into the model card on the Hub.

Pushes to `main` that touch `ml/**` also start this workflow. Pull requests that touch `ml/**` or `serve/**` run **ML smoke** (10 train steps plus `eval.py`) and fail if eval exits non-zero.

### 5. Pull the model and serve it

After the Hub upload:

```bash
pip install -r serve/requirements-serve.txt
HF_MODEL_ID=c2pig/terminal-shell-mini uvicorn serve.app:app --host 127.0.0.1 --port 8000
```

Or point at a local smoke checkpoint:

```bash
MODEL_PATH=ml/outputs/smoke uvicorn serve.app:app --host 127.0.0.1 --port 8000
```

`MODEL_PATH` wins over `HF_MODEL_ID`. The default id is `c2pig/terminal-shell-mini`. The server exposes `POST /v1/chat/completions` (non-streaming) and `GET /health`.

Check it:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"terminal-shell-mini","messages":[{"role":"user","content":"user@dev:~$ pwd"}]}'
```

### 6. Run `terminal-gpt` as a fake shell

```bash
cp .env.example .env
```

Set:

```bash
OPENAI_API_KEY=dummy
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_MODEL=terminal-shell-mini
```

Then:

```bash
terminal-gpt
```

Try `user@dev:~$ pwd` or a short question. `/exit` quits. One shot: `terminal-gpt 'user@dev:~$ whoami'`.

## Layout

| Path | Role |
| --- | --- |
| `src/terminal_gpt_mini/` | OpenAI-compatible CLI |
| `ml/data/generate_sessions.py` | Docker session capture |
| `ml/data/sample/` | Small committed JSONL |
| `ml/train.py`, `ml/eval.py` | Train and eval |
| `ml/config/tiny.yaml` | Width, steps, smoke vs full |
| `serve/app.py` | `POST /v1/chat/completions` |
| `.github/workflows/ml-smoke.yml` | PR smoke train + eval |
| `.github/workflows/ml-train.yml` | CPU train and Hub upload |
