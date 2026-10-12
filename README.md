# terminal-gpt-mini

Minimal terminal chat for [OpenAI-compatible](https://platform.openai.com/docs/api-reference/chat) APIs.

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

Requires Python 3.11+.
