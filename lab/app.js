/**
 * 4-wing knowledge museum for terminal-shell-mini
 */

const TRAIN_STEPS = [
  { id: "overview", num: "0", label: "Big picture" },
  { id: "jsonl", num: "1", label: "JSONL row" },
  { id: "tokenizer", num: "2", label: "Tokenizer" },
  { id: "encode", num: "3", label: "encode_example" },
  { id: "model", num: "4", label: "GPT-2 model" },
  { id: "batch", num: "5", label: "Batch & pad" },
  { id: "step", num: "6", label: "One train step" },
  { id: "loop", num: "7", label: "Trainer loop" },
  { id: "save", num: "8", label: "Save" },
];

const INFER_STEPS = [
  { id: "prompt", label: "Prompt & context" },
  { id: "ids", label: "IDs ≠ vectors" },
  { id: "generate", label: "Generate loop" },
  { id: "compare", label: "Train vs infer" },
];

const GLOSSARY = [
  { term: "Token ID", def: "Integer index into the vocabulary. Not a vector — embedding lookup turns it into a vector inside the model." },
  { term: "Weight / parameter", def: "A learnable number stored in model.safetensors. Updated each training step." },
  { term: "Random weights", def: "Initial values before training. build_model() starts here, not from OpenAI GPT-2." },
  { term: "Open weights", def: "Checkpoint files published on the Hub (e.g. c2pig/terminal-shell-mini), not a math term." },
  { term: "safetensors", def: "File format for named tensors (weights). Needs config.json + tokenizer + transformers code to run." },
  { term: "Context (inference)", def: "The growing sequence of input_ids: prompt + each generated token." },
  { term: "Label -100", def: "Ignore this position in loss during training only." },
  { term: "BPE", def: "Byte-pair encoding; tokenizer vocab fitted on JSONL message text once per train run." },
];

const TOUR = [
  { wing: "infer", sub: "ids", text: "Token IDs are integers, not vectors. Vectors appear after embedding." },
  { wing: "infer", sub: "generate", text: "Context at inference is prompt IDs plus one new ID per generate step." },
  { wing: "artifacts", sub: "bundle", text: "The runnable model is config + tokenizer + safetensors + library code." },
  { wing: "train", sub: "tokenizer", text: "Tokenizer is fit once on JSONL text; it does not update during trainer.train()." },
  { wing: "train", sub: "model", text: "Weights start random; training rewrites safetensors." },
  { wing: "map", sub: "overview", text: "Docker data never runs at inference — only tokens built from chat text." },
];

const state = {
  wing: "map",
  sub: "overview",
  example: null,
  repoMap: null,
  trace: null,
  tensors: null,
  tourIndex: -1,
  inferStepper: 0,
  mapNode: "train",
};

function el(id) {
  return document.getElementById(id);
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

async function loadJson(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load ${path}`);
  return res.json();
}

function setBreadcrumb() {
  const wing = state.wing.toUpperCase();
  let sub = state.sub;
  if (state.wing === "train") {
    const s = TRAIN_STEPS.find((x) => x.id === sub);
    sub = s ? s.label : sub;
  } else if (state.wing === "infer") {
    const s = INFER_STEPS.find((x) => x.id === sub);
    sub = s ? s.label : sub;
  }
  el("breadcrumb").textContent = `${wing} › ${sub}`;
}

function activateWing(wing, sub) {
  state.wing = wing;
  state.sub = sub || (wing === "train" ? "overview" : wing === "infer" ? "prompt" : wing === "map" ? "overview" : "bundle");

  document.querySelectorAll(".wing-btn").forEach((b) => b.classList.toggle("active", b.dataset.wing === wing));
  document.querySelectorAll(".wing-panel").forEach((p) => p.classList.toggle("active", p.dataset.wing === wing));

  const subNav = el("sub-nav");
  subNav.innerHTML = "";
  subNav.classList.toggle("hidden", wing === "map" && false);

  if (wing === "train") {
    subNav.classList.remove("hidden");
    subNav.innerHTML = `<h2>Steps</h2>` + TRAIN_STEPS.map(
      (s) =>
        `<button type="button" class="step-btn" data-train-step="${s.id}"><span class="num">${s.num}</span><span>${s.label}</span></button>`
    ).join("");
    subNav.querySelectorAll("[data-train-step]").forEach((btn) => {
      btn.addEventListener("click", () => activateTrainStep(btn.dataset.trainStep));
    });
    activateTrainStep(state.sub);
  } else if (wing === "infer") {
    subNav.classList.remove("hidden");
    subNav.innerHTML = `<h2>Infer</h2>` + INFER_STEPS.map(
      (s) => `<button type="button" class="step-btn" data-infer-step="${s.id}"><span>${s.label}</span></button>`
    ).join("");
    subNav.querySelectorAll("[data-infer-step]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.sub = btn.dataset.inferStep;
        renderInfer();
        setBreadcrumb();
        syncHash();
        subNav.querySelectorAll("[data-infer-step]").forEach((b) => b.classList.toggle("active", b.dataset.inferStep === state.sub));
      });
    });
    renderInfer();
    subNav.querySelectorAll("[data-infer-step]").forEach((b) => b.classList.toggle("active", b.dataset.inferStep === state.sub));
  } else if (wing === "artifacts") {
    subNav.classList.add("hidden");
    renderArtifacts();
  } else if (wing === "map") {
    subNav.classList.add("hidden");
    renderMap();
  }

  setBreadcrumb();
  syncHash();
}

function activateTrainStep(stepId) {
  state.sub = stepId;
  const train = el("wing-train");
  train.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.dataset.step === stepId));
  el("sub-nav").querySelectorAll("[data-train-step]").forEach((b) => {
    b.classList.toggle("active", b.dataset.trainStep === stepId);
  });
  setBreadcrumb();
  syncHash();
}

function syncHash() {
  const h = `${state.wing}/${state.sub}`;
  if (location.hash !== `#${h}`) history.replaceState(null, "", `#${h}`);
}

function parseHash() {
  const raw = (location.hash || "#map/overview").replace("#", "");
  const [wing, sub] = raw.split("/");
  if (["map", "train", "artifacts", "infer"].includes(wing)) {
    activateWing(wing, sub || undefined);
  }
}

/* ——— MAP ——— */
function renderMap() {
  const root = el("wing-map");
  const rm = state.repoMap;
  if (!rm) {
    root.innerHTML = "<p>Loading map…</p>";
    return;
  }
  const flowOrder = ["docker", "generate", "jsonl", "train", "hub", "serve", "space", "cli"];
  const ordered = flowOrder.map((id) => rm.nodes.find((n) => n.id === id)).filter(Boolean);
  const nodeBtns = ordered
    .map((n, i) => {
      const arrow = i < ordered.length - 1 ? '<div style="text-align:center;color:var(--muted)">↓</div>' : "";
      return `<button type="button" class="map-node-btn ${state.mapNode === n.id ? "active" : ""}" data-map-node="${n.id}">${escapeHtml(n.label)}</button>${arrow}`;
    })
    .join("");

  const selected = rm.nodes.find((n) => n.id === state.mapNode) || rm.nodes[0];
  root.innerHTML = `
    <h2>MAP — Repo flow</h2>
    <p class="lead">Click a node. Inference paths only load Hub weights + tokenizer — not Docker or JSONL.</p>
    <div class="card map-flow">${nodeBtns}</div>
    <div class="map-detail" id="map-detail"></div>
    <div class="card" style="margin-top:0.75rem;font-size:0.88rem;color:var(--muted)">
      ${rm.facts.map((f) => `<p style="margin:0.35rem 0">• ${escapeHtml(f)}</p>`).join("")}
    </div>
    <details class="drill"><summary>Double-click — jump wings</summary>
      <div class="drill-body">
        <button type="button" class="btn" data-jump="train/overview">TRAIN</button>
        <button type="button" class="btn" data-jump="artifacts/bundle">ARTIFACTS</button>
        <button type="button" class="btn" data-jump="infer/prompt">INFER</button>
      </div>
    </details>`;

  root.querySelectorAll("[data-map-node]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.mapNode = btn.dataset.mapNode;
      renderMap();
    });
  });
  root.querySelectorAll("[data-jump]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const [w, s] = btn.dataset.jump.split("/");
      activateWing(w, s);
    });
  });

  const edges = rm.edges.filter((e) => e.from === selected.id || e.to === selected.id);
  el("map-detail").innerHTML = `
    <strong>${escapeHtml(selected.label)}</strong>
    ${selected.path ? `<div style="font-family:var(--mono);font-size:0.8rem;margin:0.35rem 0">${escapeHtml(selected.path)}</div>` : ""}
    <p style="margin:0.5rem 0 0;color:var(--muted)">Layer: ${escapeHtml(selected.layer)}</p>
    ${edges.length ? `<ul style="margin:0.5rem 0 0;padding-left:1.2rem">${edges.map((e) => `<li>${escapeHtml(e.from)} → ${escapeHtml(e.to)}: ${escapeHtml(e.note)}</li>`).join("")}</ul>` : ""}`;
}

/* ——— ARTIFACTS ——— */
let artifactCard = "safetensors";

function renderArtifacts() {
  const root = el("wing-artifacts");
  const tensors = state.tensors || [];
  root.innerHTML = `
    <h2>ARTIFACTS — What “the model” is on disk</h2>
    <p class="lead">Three files + Hugging Face <code>transformers</code> code. Safetensors alone is not runnable.</p>
    <div class="bundle-row">
      <div class="bundle-card ${artifactCard === "config" ? "active" : ""}" data-card="config"><strong>config.json</strong><br/><span style="color:var(--muted);font-size:0.8rem">Architecture recipe</span></div>
      <div class="bundle-card ${artifactCard === "tokenizer" ? "active" : ""}" data-card="tokenizer"><strong>tokenizer.*</strong><br/><span style="color:var(--muted);font-size:0.8rem">Text ↔ IDs</span></div>
      <div class="bundle-card ${artifactCard === "safetensors" ? "active" : ""}" data-card="safetensors"><strong>model.safetensors</strong><br/><span style="color:var(--muted);font-size:0.8rem">Learned weights</span></div>
    </div>
    <div class="card" id="artifact-detail"></div>
    ${artifactCard === "safetensors" && tensors.length ? `<div class="card"><h3>Named tensors (smoke checkpoint)</h3><div class="tensor-list" id="tensor-list"></div></div>` : ""}
    <details class="drill"><summary>Double-click — open weights</summary><div class="drill-body">Publishing <code>c2pig/terminal-shell-mini</code> on the Hub means sharing these files under a license — “open weights,” not “open source” for code.</div></details>`;

  root.querySelectorAll("[data-card]").forEach((c) => {
    c.addEventListener("click", () => {
      artifactCard = c.dataset.card;
      renderArtifacts();
    });
  });

  const details = {
    config: "Lists n_layer, n_embd, vocab_size, n_positions. Used to construct an empty GPT2LMHeadModel before loading weights.",
    tokenizer: "Built once in build_tokenizer() from JSONL text. Saved beside weights; inference only encodes/decodes — no refit.",
    safetensors: "Binary store of named tensors (embedding tables, attention matrices, …). ~3.4M parameters total.",
  };
  el("artifact-detail").innerHTML = `<p style="margin:0">${details[artifactCard]}</p>`;

  if (artifactCard === "safetensors" && tensors.length) {
    const list = el("tensor-list");
    if (list) {
      list.innerHTML = tensors
        .slice(0, 40)
        .map((t) => `<div class="tensor-row" title="${escapeHtml(t.name)}">${escapeHtml(t.name)} <span style="color:var(--muted)">[${t.shape.join("×")}]</span></div>`)
        .join("");
    }
  }
}

/* ——— INFER ——— */
function renderInfer() {
  const root = el("wing-infer");
  const ex = state.example;
  const trace = state.trace;
  const sub = state.sub;

  if (sub === "prompt") {
    root.innerHTML = `
      <h2>INFER — Prompt &amp; context</h2>
      <p class="lead">serve/, Space, and eval build the same chat string as training, then tokenize. <strong>Context</strong> = that ID sequence, growing during <code>generate()</code>.</p>
      <div class="card"><h3>Messages (pwd example)</h3><div class="chat" id="infer-chat"></div></div>
      <div class="card"><h3>Rendered prompt (before generation)</h3><pre class="io-box io-in">${ex ? escapeHtml(ex.rendered.split("<|assistant|>")[0] + "<|assistant|>\n") : "—"}</pre></div>
      <details class="drill"><summary>Double-click — code paths</summary><div class="drill-body"><code>ml/chat_format.py</code> → <code>prompt_token_ids</code> in eval, serve, space/app.py.</div></details>`;
    if (ex) renderChat(ex.messages, "infer-chat");
    return;
  }

  if (sub === "ids") {
    const step = state.inferStepper;
    const steps = ["Text", "Token IDs", "Embeddings", "Logits"];
    root.innerHTML = `
      <h2>INFER — IDs ≠ vectors</h2>
      <p class="lead">The model never receives strings. It receives integers, then embeddings (vectors).</p>
      <div class="stepper-tabs">${steps.map((s, i) => `<button type="button" class="stepper-tab ${i === step ? "active" : ""}" data-stepper="${i}">${i + 1}. ${s}</button>`).join("")}</div>
      <div class="card" id="stepper-body"></div>`;
    root.querySelectorAll("[data-stepper]").forEach((b) => {
      b.addEventListener("click", () => {
        state.inferStepper = Number(b.dataset.stepper);
        renderInfer();
      });
    });
    const body = el("stepper-body");
    const nEmb = trace?.n_embd || 256;
    const vocab = trace?.vocab_size || 800;
    if (step === 0) {
      body.innerHTML = `<p>User types <code>pwd</code> → formatted as <code>user@dev:~$ pwd</code> inside <code>&lt;|user|&gt;</code> block.</p>`;
    } else if (step === 1) {
      const ids = trace?.prompt_ids?.slice(0, 24) || ex?.encoded?.input_ids?.slice(0, 24) || [];
      body.innerHTML = `<p><strong>These are indices</strong> (not vectors):</p><div class="token-strip">${ids.map((id) => `<span class="token">${id}</span>`).join("")}${ids.length ? " …" : ""}</div>`;
    } else if (step === 2) {
      body.innerHTML = `<p>Each ID looks up a row in the embedding table (${vocab} × ${nEmb} weights in safetensors).</p><div class="embed-bar">${Array.from({ length: 24 }, (_, i) => `<span style="height:${12 + (i % 5) * 4}px"></span>`).join("")}</div><p style="font-size:0.85rem;color:var(--muted)">Schematic: ${nEmb} floats per token position.</p>`;
    } else {
      body.innerHTML = `<p>At each position, LM head outputs <strong>${vocab} scores</strong> (logits) — one per vocab entry. Softmax → pick next token.</p>`;
    }
    return;
  }

  if (sub === "generate") {
    const steps = trace?.steps || [];
    const idx = Math.min(state.inferGenStep || 0, Math.max(0, steps.length - 1));
    state.inferGenStep = idx;
    const promptLen = trace?.prompt_len || 0;
    let ctxHtml = `<span class="ctx-prompt">prompt (${promptLen} ids)</span>`;
    if (steps[idx]) {
      ctxHtml += ` <span class="ctx-gen">+ ${idx + 1} generated → "${escapeHtml(steps[idx].decoded_new)}"</span>`;
    }
    root.innerHTML = `
      <h2>INFER — Generate loop</h2>
      <p class="lead">Each step appends one token ID to context (causal: only left tokens visible).</p>
      <div class="context-bar">${ctxHtml}</div>
      <div class="slider-row" style="display:flex;align-items:center;gap:0.75rem">
        <button type="button" class="btn" id="gen-prev">◀</button>
        <span>Step ${steps.length ? idx + 1 : 0} / ${steps.length}</span>
        <button type="button" class="btn" id="gen-next">▶</button>
      </div>
      <div class="card">${steps[idx] ? `<p>New token ID: <code>${steps[idx].new_id}</code> piece: <code>${escapeHtml(steps[idx].piece)}</code></p><p>Decoded assistant so far: <strong>${escapeHtml(steps[idx].decoded_new)}</strong></p>` : "<p>Run export_museum_data.py with smoke checkpoint for live trace.</p>"}</div>`;
    el("gen-prev")?.addEventListener("click", () => {
      state.inferGenStep = Math.max(0, idx - 1);
      renderInfer();
    });
    el("gen-next")?.addEventListener("click", () => {
      state.inferGenStep = Math.min(steps.length - 1, idx + 1);
      renderInfer();
    });
    return;
  }

  if (sub === "compare") {
    root.innerHTML = `
      <h2>INFER — Train vs infer</h2>
      <table class="params compare-table">
        <tr><th></th><th>Training (train.py)</th><th>Inference (serve / Space)</th></tr>
        <tr><td>chat_format</td><td>encode_example</td><td>prompt_token_ids + decode_new_tokens</td></tr>
        <tr><td>labels / -100</td><td>Yes</td><td>No</td></tr>
        <tr><td>Assistant text</td><td>In batch (supervision)</td><td>Model generates</td></tr>
        <tr><td>Tokenizer updates</td><td>No (fit once at start)</td><td>No (load from disk)</td></tr>
        <tr><td>Output</td><td>safetensors on disk</td><td>Text to user</td></tr>
      </table>`;
  }
}

/* ——— TRAIN data fills ——— */
function renderChat(messages, containerId) {
  const root = el(containerId);
  if (!root) return;
  root.innerHTML = messages
    .map(
      (m) =>
        `<div class="bubble bubble-${m.role}"><div class="role">${m.role}</div><div>${escapeHtml(m.content)}</div></div>`
    )
    .join("");
}

function fillTrainData() {
  const ex = state.example;
  if (!ex) return;
  renderChat(ex.messages, "chat-jsonl");
  el("raw-json").textContent = JSON.stringify({ messages: ex.messages }, null, 2);
  el("rendered-text").textContent = ex.rendered;
  renderTokenStrip();
  renderAlignTable();
  fillYamlTable();
  initBatchDemo();
  initTrainStepDemo();
  initLoopSlider();
}

function renderTokenStrip() {
  const { encoded, token_spans } = state.example;
  const labels = encoded.labels;
  el("token-strip").innerHTML = token_spans
    .map((t, i) => {
      const train = labels[i] !== -100;
      return `<span class="token ${train ? "train" : "ignore"}" title="id=${t.id}">${escapeHtml(t.piece.replace(/\n/g, "↵"))}</span>`;
    })
    .join("");
  el("encode-stats").textContent = `${encoded.input_ids.length} tokens · ${labels.filter((l) => l !== -100).length} trained`;
}

function renderAlignTable() {
  const { encoded, token_spans } = state.example;
  const show = Math.min(encoded.input_ids.length, 28);
  let rows = "";
  for (let i = 0; i < show; i++) {
    const train = encoded.labels[i] !== -100;
    rows += `<tr><td>${i}</td><td>${encoded.input_ids[i]}</td><td><code>${escapeHtml(token_spans[i].piece.replace(/\n/g, "\\n"))}</code></td><td style="color:${train ? "var(--train)" : "var(--ignore)"}">${encoded.labels[i]}</td></tr>`;
  }
  el("align-tbody").innerHTML = rows;
}

function fillYamlTable() {
  const y = state.example.yaml_snippet;
  el("yaml-table").innerHTML = `
    <tr><th>n_layer</th><td>${y.n_layer}</td><td>Transformer blocks</td></tr>
    <tr><th>n_embd</th><td>${y.n_embd}</td><td>Vector size per token</td></tr>
    <tr><th>vocab_size</th><td>~${y.vocab_size}</td><td>Actual: ${state.example.vocab_size}</td></tr>
    <tr><th>batch_size</th><td>${y.batch_size}</td><td>Per step</td></tr>`;
}

function initBatchDemo() {
  const demos = [
    { ids: [10, 11, 12], labels: [-100, 42, 43] },
    { ids: [20, 21], labels: [-100, 55] },
  ];
  const padId = 0;
  const width = 3;
  let html = "<table class='pad-demo'><tr><th></th>";
  for (let c = 0; c < width; c++) html += `<th>${c}</th>`;
  html += "</tr>";
  for (const ex of demos) {
    html += "<tr><td>ids</td>";
    for (let c = 0; c < width; c++) html += `<td>${c < ex.ids.length ? ex.ids[c] : `<span class='pad'>${padId}</span>`}</td>`;
    html += "</tr><tr><td>labels</td>";
    for (let c = 0; c < width; c++) html += `<td>${c < ex.labels.length ? ex.labels[c] : "<span class='pad'>-100</span>"}</td>`;
    html += "</tr>";
  }
  html += "</table>";
  el("pad-demo").innerHTML = html;
}

function initTrainStepDemo() {
  const { encoded, token_spans } = state.example;
  let idx = encoded.input_ids.length - 1;
  while (idx > 0 && encoded.labels[idx] === -100) idx -= 1;
  el("train-step-detail").innerHTML = `<p>Position <strong>${idx}</strong>: predict token <code>${encoded.labels[idx]}</code> (${escapeHtml(token_spans[idx].piece)}) given all prior tokens.</p>`;
}

function initLoopSlider() {
  const slider = el("step-slider");
  const bar = el("loss-bar-inner");
  function update() {
    const v = Number(slider.value);
    el("step-readout").textContent = `Step ${v} / 500 (smoke)`;
    const fakeLoss = (2.8 * Math.exp(-v / 180) + 0.35).toFixed(3);
    el("fake-loss").textContent = fakeLoss;
    bar.style.width = `${Math.min(100, (1 - parseFloat(fakeLoss) / 3) * 100)}%`;
  }
  slider.oninput = update;
  update();
}

/* Glossary & tour */
function openGlossary() {
  el("glossary-body").innerHTML = GLOSSARY.map((g) => `<dl class="glossary-item"><dt>${escapeHtml(g.term)}</dt><dd>${escapeHtml(g.def)}</dd></dl>`).join("");
  el("glossary-modal").classList.add("open");
}

function startTour() {
  state.tourIndex = 0;
  el("tour-bar").classList.add("visible");
  renderTour();
}

function renderTour() {
  const t = TOUR[state.tourIndex];
  el("tour-text").textContent = `Tour ${state.tourIndex + 1}/${TOUR.length}: ${t.text}`;
  el("tour-dots").innerHTML = TOUR.map((_, i) => `<span class="${i < state.tourIndex ? "done" : ""} ${i === state.tourIndex ? "current" : ""}"></span>`).join("");
  activateWing(t.wing, t.sub);
}

function initUi() {
  document.querySelectorAll(".wing-btn").forEach((b) => {
    b.addEventListener("click", () => activateWing(b.dataset.wing));
  });
  el("btn-glossary").addEventListener("click", openGlossary);
  el("glossary-close").addEventListener("click", () => el("glossary-modal").classList.remove("open"));
  el("glossary-modal").addEventListener("click", (e) => {
    if (e.target === el("glossary-modal")) el("glossary-modal").classList.remove("open");
  });
  el("btn-tour").addEventListener("click", startTour);
  el("tour-skip").addEventListener("click", () => {
    state.tourIndex = -1;
    el("tour-bar").classList.remove("visible");
  });
  el("tour-next").addEventListener("click", () => {
    if (state.tourIndex < TOUR.length - 1) {
      state.tourIndex += 1;
      renderTour();
    } else {
      state.tourIndex = -1;
      el("tour-bar").classList.remove("visible");
    }
  });
  window.addEventListener("hashchange", parseHash);
}

async function init() {
  initUi();
  try {
    state.example = await loadJson("data/pwd-example.json");
    state.repoMap = await loadJson("data/repo-map.json");
    try {
      state.trace = await loadJson("data/inference-trace.json");
    } catch {
      state.trace = null;
    }
    try {
      state.tensors = await loadJson("data/tensor-manifest.json");
    } catch {
      state.tensors = [];
    }
    fillTrainData();
    parseHash();
  } catch (e) {
    el("load-error").hidden = false;
    el("load-error").textContent = e.message;
    activateWing("map", "overview");
  }
}

init();
