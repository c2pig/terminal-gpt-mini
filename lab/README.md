# Training lab (interactive)

Static single-page app that walks through **`ml/train.py`** step by step, with expandable “double-click” detail sections.

## Run locally

Browsers block `fetch()` on `file://`, so serve the repo over HTTP:

```bash
cd /path/to/terminal-gpt-mini
python3 -m http.server 8080
```

Open **http://localhost:8080/lab/**

## Regenerate example data

Uses a real `encode_example()` pass and the smoke checkpoint tokenizer (or builds BPE like `train.py`):

```bash
python3 lab/scripts/export_example.py
```

Writes `lab/data/pwd-example.json`.

## Files

| File | Purpose |
|------|---------|
| `index.html` | Step panels + drill-down copy |
| `styles.css` | Layout and token colors |
| `app.js` | Navigation, loads JSON, renders tables |
| `data/pwd-example.json` | Precomputed tokens/labels for one `pwd` session |
