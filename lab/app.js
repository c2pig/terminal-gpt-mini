/**
 * terminal-shell-mini training lab — loads real encode_example output from lab/data/.
 */

const STEPS = [
  { id: "overview", num: "0", label: "Big picture", hint: "Inputs → train.py → outputs" },
  { id: "jsonl", num: "1", label: "JSONL row", hint: "One conversation" },
  { id: "tokenizer", num: "2", label: "Tokenizer", hint: "Text → token IDs" },
  { id: "encode", num: "3", label: "encode_example", hint: "input_ids + labels" },
  { id: "model", num: "4", label: "GPT-2 model", hint: "~3.4M parameters" },
  { id: "batch", num: "5", label: "Batch & pad", hint: "PadCollator" },
  { id: "step", num: "6", label: "One train step", hint: "Loss on assistant only" },
  { id: "loop", num: "7", label: "Trainer loop", hint: "max_steps × AdamW" },
  { id: "save", num: "8", label: "Save", hint: "Files on disk" },
];

let example = null;

async function loadExample() {
  const res = await fetch("data/pwd-example.json");
  if (!res.ok) throw new Error("Could not load data/pwd-example.json — serve via HTTP (see lab/README.md)");
  example = await res.json();
}

function el(id) {
  return document.getElementById(id);
}

function renderNav() {
  const nav = el("steps-nav-list");
  nav.innerHTML = STEPS.map(
    (s) => `
    <button type="button" class="step-btn" data-step="${s.id}" aria-current="false">
      <span class="num">${s.num}</span>
      <span>
        <span class="label">${s.label}</span>
        <span class="hint">${s.hint}</span>
      </span>
    </button>`
  ).join("");

  nav.querySelectorAll(".step-btn").forEach((btn) => {
    btn.addEventListener("click", () => activateStep(btn.dataset.step));
  });
}

function activateStep(id) {
  document.querySelectorAll(".step-btn").forEach((b) => {
    const on = b.dataset.step === id;
    b.classList.toggle("active", on);
    b.setAttribute("aria-current", on ? "step" : "false");
  });
  document.querySelectorAll(".panel").forEach((p) => {
    p.classList.toggle("active", p.dataset.step === id);
  });
  if (location.hash !== `#${id}`) {
    history.replaceState(null, "", `#${id}`);
  }
}

function renderChat(messages, containerId) {
  const root = el(containerId);
  root.innerHTML = messages
    .map((m) => {
      const role = m.role;
      return `<div class="bubble bubble-${role}">
        <div class="role">${role}</div>
        <div>${escapeHtml(m.content)}</div>
      </div>`;
    })
    .join("");
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderTokenStrip() {
  const { encoded, token_spans } = example;
  const labels = encoded.labels;
  const strip = el("token-strip");
  strip.innerHTML = token_spans
    .map((t, i) => {
      const train = labels[i] !== -100;
      const cls = train ? "train" : "ignore";
      const title = `pos ${i} id=${t.id} label=${labels[i]}`;
      const piece = t.piece.replace(/\n/g, "↵");
      return `<span class="token ${cls}" title="${escapeHtml(title)}">${escapeHtml(piece)}</span>`;
    })
    .join("");

  el("encode-stats").textContent = `${encoded.input_ids.length} tokens · ${
    labels.filter((l) => l !== -100).length
  } trained (label ≠ -100)`;
}

function renderAlignTable() {
  const { encoded, token_spans } = example;
  const tbody = el("align-tbody");
  const show = Math.min(encoded.input_ids.length, 24);
  let rows = "";
  for (let i = 0; i < show; i++) {
    const train = encoded.labels[i] !== -100;
    rows += `<tr>
      <td>${i}</td>
      <td>${encoded.input_ids[i]}</td>
      <td><code>${escapeHtml(token_spans[i].piece.replace(/\n/g, "\\n"))}</code></td>
      <td style="color:${train ? "var(--train)" : "var(--ignore)"}">${encoded.labels[i]}</td>
    </tr>`;
  }
  if (encoded.input_ids.length > show) {
    rows += `<tr><td colspan="4" style="color:var(--muted)">… ${encoded.input_ids.length - show} more tokens</td></tr>`;
  }
  tbody.innerHTML = rows;
}

function initBatchDemo() {
  const demos = [
    { name: "ex A", ids: [10, 11, 12], labels: [-100, 42, 43] },
    { name: "ex B", ids: [20, 21], labels: [-100, 55] },
  ];
  const padId = 0;
  const width = Math.max(...demos.map((d) => d.ids.length));
  let html = "<table class='pad-demo'><tr><th></th>";
  for (let c = 0; c < width; c++) html += `<th>${c}</th>`;
  html += "</tr>";

  for (const ex of demos) {
    html += `<tr><td>input_ids</td>`;
    for (let c = 0; c < width; c++) {
      if (c < ex.ids.length) html += `<td>${ex.ids[c]}</td>`;
      else html += `<td class="pad">${padId}</td>`;
    }
    html += "</tr><tr><td>labels</td>";
    for (let c = 0; c < width; c++) {
      if (c < ex.labels.length) html += `<td>${ex.labels[c]}</td>`;
      else html += `<td class="pad">-100</td>`;
    }
    html += "</tr>";
  }
  html += "</table>";
  el("pad-demo").innerHTML = html;
}

function initTrainStepDemo() {
  const position = 40;
  const { encoded, token_spans } = example;
  let idx = Math.min(position, encoded.input_ids.length - 1);
  while (idx > 0 && encoded.labels[idx] === -100) idx -= 1;
  const targetId = encoded.labels[idx];
  const ctx = token_spans
    .slice(Math.max(0, idx - 5), idx)
    .map((t) => t.piece.replace(/\n/g, "↵"))
    .join(" ");
  el("train-step-detail").innerHTML = `
    <p>At position <strong>${idx}</strong>, the model sees all tokens before this point (causal mask), then predicts token ID <strong>${targetId}</strong>
    (<code>${escapeHtml(token_spans[idx].piece)}</code>).</p>
    <p class="io-box io-in" style="margin-top:0.5rem">…context: ${escapeHtml(ctx)} → <strong>?</strong></p>
    <p style="margin:0.5rem 0 0;font-size:0.85rem;color:var(--muted)">Cross-entropy compares the model’s softmax over the vocab to the true next token. Positions with label <code>-100</code> are skipped.</p>`;
}

function initLoopSlider() {
  const slider = el("step-slider");
  const out = el("step-readout");
  const bar = el("loss-bar-inner");
  const max = 500;
  slider.max = max;
  slider.value = 120;
  function update() {
    const v = Number(slider.value);
    out.textContent = `Step ${v} / ${max} (smoke default in tiny.yaml)`;
    const fakeLoss = (2.8 * Math.exp(-v / 180) + 0.35).toFixed(3);
    el("fake-loss").textContent = fakeLoss;
    bar.style.width = `${Math.min(100, (1 - parseFloat(fakeLoss) / 3) * 100)}%`;
  }
  slider.addEventListener("input", update);
  update();
}

function fillStatic() {
  const y = example.yaml_snippet;
  el("yaml-table").innerHTML = `
    <tr><th>n_layer</th><td>${y.n_layer}</td><td>Transformer blocks stacked</td></tr>
    <tr><th>n_embd</th><td>${y.n_embd}</td><td>Vector size per token</td></tr>
    <tr><th>n_head</th><td>${y.n_head}</td><td>Attention heads (${y.n_embd / y.n_head} dims each)</td></tr>
    <tr><th>n_inner</th><td>${y.n_inner}</td><td>Feed-forward hidden width</td></tr>
    <tr><th>n_positions</th><td>${y.n_positions}</td><td>Max sequence length</td></tr>
    <tr><th>vocab_size</th><td>~${y.vocab_size}</td><td>BPE tokens (actual: ${example.vocab_size})</td></tr>
    <tr><th>batch_size</th><td>${y.batch_size}</td><td>Examples per optimizer step</td></tr>
    <tr><th>max_steps (smoke)</th><td>${y.max_steps_smoke}</td><td>How many updates</td></tr>`;

  el("rendered-text").textContent = example.rendered;
  el("raw-json").textContent = JSON.stringify({ messages: example.messages }, null, 2);
}

async function init() {
  try {
    await loadExample();
  } catch (e) {
    el("load-error").hidden = false;
    el("load-error").textContent = e.message;
    return;
  }

  renderNav();
  fillStatic();
  renderChat(example.messages, "chat-jsonl");
  renderTokenStrip();
  renderAlignTable();
  initBatchDemo();
  initTrainStepDemo();
  initLoopSlider();

  const hash = location.hash.replace("#", "");
  activateStep(STEPS.some((s) => s.id === hash) ? hash : "overview");
  window.addEventListener("hashchange", () => {
    const id = location.hash.replace("#", "");
    if (STEPS.some((s) => s.id === id)) activateStep(id);
  });
}

init();
