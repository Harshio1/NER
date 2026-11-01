"use strict";
const $ = s => document.querySelector(s);
const TOKEN_RE = /[A-Za-z0-9]+(?:[.'’-][A-Za-z0-9]+)*/g;
let M = null, R = null, DS = [];

/* ---------- BiGRU forward pass (mirror of src/gru_numpy.py) ---------- */
const sig = x => 1 / (1 + Math.exp(-x));
function gru(X, W, Uzr, Un, b, H) {
  const out = []; let h = new Float64Array(H);
  for (const x of X) {
    const xw = new Float64Array(3 * H);
    for (let j = 0; j < 3 * H; j++) { let s = b[j]; for (let d = 0; d < x.length; d++) s += x[d] * W[d][j]; xw[j] = s; }
    const z = new Float64Array(H), r = new Float64Array(H), nh = new Float64Array(H);
    for (let j = 0; j < H; j++) {
      let a = xw[j], c = xw[H + j];
      for (let k = 0; k < H; k++) { a += h[k] * Uzr[k][j]; c += h[k] * Uzr[k][H + j]; }
      z[j] = sig(a); r[j] = sig(c);
    }
    for (let j = 0; j < H; j++) {
      let a = xw[2 * H + j];
      for (let k = 0; k < H; k++) a += r[k] * h[k] * Un[k][j];
      nh[j] = (1 - z[j]) * Math.tanh(a) + z[j] * h[j];
    }
    h = nh; out.push(h);
  }
  return out;
}
function tagSentence(text) {
  const toks = text.match(TOKEN_RE) || [];
  if (!toks.length) return { toks, tags: [], conf: [] };
  const P = M.params, H = M.hidden;
  const w2i = M.w2i, s2i = M.s2i;
  const X = toks.map((t, i) => {
    const lw = t.toLowerCase(), wid = w2i[lw] ?? 1, sid = s2i[lw.slice(-3)] ?? 1;
    return [...P.Ew[wid], ...P.Es[sid], (i > 0 && t[0] !== t[0].toLowerCase()) ? 1 : 0, i === 0 ? 1 : 0];
  });
  const hf = gru(X, P.Wf, P.Uzrf, P.Unf, P.bf, H);
  const hb = gru([...X].reverse(), P.Wb, P.Uzrb, P.Unb, P.bb, H).reverse();
  const tags = [], conf = [];
  toks.forEach((_, i) => {
    const h = [...hf[i], ...hb[i]], lg = P.bo.map((b, k) => h.reduce((s, v, d) => s + v * P.Wo[d][k], b));
    const m = Math.max(...lg), e = lg.map(v => Math.exp(v - m)), sum = e.reduce((a, b) => a + b);
    const k = e.indexOf(Math.max(...e)); tags.push(M.tags[k]); conf.push(e[k] / sum);
  });
  return { toks, tags, conf };
}
function spans(tags) {
  const out = []; let s = null, t = null;
  [...tags, "O"].forEach((g, i) => {
    if (g === "O" || g[0] === "B" || (g[0] === "I" && g.slice(2) !== t)) { if (s !== null) out.push([t, s, i]); s = t = null; }
    if (g !== "O" && s === null) { s = i; t = g.slice(2); }
  });
  return out;
}

/* ---------- rendering ---------- */
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.textContent = html; return e; };
const typ = g => g === "O" ? "O" : g.slice(2);
function tokens(toks, tags, ref) {
  const box = el("div", "tagged");
  toks.forEach((t, i) => {
    const c = el("span", `tok ${typ(tags[i])}` + (ref && ref[i] !== tags[i] ? " wrong" : ""));
    c.append(el("b", "", t), el("i", "", tags[i])); box.append(c);
  });
  return box;
}
function run() {
  const { toks, tags } = tagSentence($("#inp").value);
  $("#out").replaceWith(Object.assign(tokens(toks, tags), { id: "out" }));
  const sp = spans(tags), ents = $("#ents"); ents.textContent = "";
  if (!sp.length) ents.textContent = toks.length ? "No entities found." : "Type or say a sentence.";
  sp.forEach(([t, a, b], i) => {
    const s = el("span", t); s.append(el("b", "", t + " "), document.createTextNode(toks.slice(a, b).join(" ") + (i < sp.length - 1 ? "   " : ""))); ents.append(s);
  });
}

/* ---------- microphone: live waveform + Web Speech API ---------- */
const cv = $("#wave"), cx = cv.getContext("2d");
let analyser, raf, rec;
function draw() {
  const w = cv.width = cv.clientWidth * devicePixelRatio, h = cv.height = 90 * devicePixelRatio;
  const ink = getComputedStyle(document.body).color; cx.clearRect(0, 0, w, h); cx.strokeStyle = ink; cx.lineWidth = 2 * devicePixelRatio; cx.beginPath();
  if (analyser) {
    const d = new Uint8Array(analyser.fftSize); analyser.getByteTimeDomainData(d);
    d.forEach((v, i) => { const x = i / d.length * w, y = h / 2 + (v - 128) / 128 * h * .9; i ? cx.lineTo(x, y) : cx.moveTo(x, y); });
  } else { cx.moveTo(0, h / 2); cx.lineTo(w, h / 2); }
  cx.stroke(); if (analyser) raf = requestAnimationFrame(draw);
}
async function toggleMic() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition, btn = $("#mic");
  if (rec) { rec.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const ac = new AudioContext(); analyser = ac.createAnalyser(); ac.createMediaStreamSource(stream).connect(analyser);
    rec = new SR(); rec.lang = "en-IN"; rec.interimResults = true;
    rec.onresult = e => { $("#inp").value = [...e.results].map(r => r[0].transcript).join(" ").trim(); run(); };
    rec.onend = () => { rec = null; analyser = null; cancelAnimationFrame(raf); stream.getTracks().forEach(t => t.stop()); ac.close(); btn.textContent = "Speak"; btn.classList.remove("live"); draw(); };
    rec.start(); btn.textContent = "Stop"; btn.classList.add("live"); draw();
    $("#status").textContent = "Listening. Speech recognition is done by your browser; tagging runs locally.";
  } catch (err) { $("#status").textContent = "Microphone unavailable (" + err.name + "). Allow access, or type a sentence instead."; }
}

/* ---------- dataset / results sections ---------- */
function showDataset() {
  const box = $("#dataset");
  DS.forEach(s => {
    const b = el("button", "ds" + (s.split === "test" ? " test" : "")); b.append(el("small", "", s.id));
    const t = el("div"); s.tokens.forEach((w, i) => { const sp = el("span", s.tags[i] === "O" ? "" : typ(s.tags[i]), w + " "); t.append(sp); });
    b.append(t); b.onclick = () => { $("#inp").value = s.tokens.join(" "); run(); location.hash = "demo"; };
    box.append(b);
  });
}
const pct = v => (v * 100).toFixed(0) + "%";
function showResults() {
  const a = R.ablation_mean_std_f1, c = R.config;
  $("#f1").textContent = pct(R.test_cased.f1); $("#devf1").textContent = pct(R.dev_unseen_entities.f1);
  $("#gain").textContent = pct(a.real_plus_augmented.test_cased.mean); $("#gain0").textContent = pct(a.real_only_10.test_cased.mean);
  const rows = [["10 recordings only", a.real_only_10, "a"], ["10 recordings + augmented", a.real_plus_augmented, "b"]];
  [["test_cased", "Test, original casing"], ["test_lower", "Test, lowercase (ASR style)"], ["dev", "Unseen synthetic entities"]].forEach(([k, label]) => rows.forEach(([n, v, cls]) => {
    const r = el("div", "bar " + cls); r.append(el("span", "", `${label}: ${n}`));
    const d = el("div"), i = el("i"); i.style.width = pct(v[k].mean); d.append(i); r.append(d, el("em", "", `${pct(v[k].mean)} ±${(v[k].std * 100).toFixed(0)}`)); $("#bars").append(r);
  }));
  Object.entries(R.test_cased.per_type).forEach(([t, m]) => {
    const tr = el("tr"); [t, pct(m.precision), pct(m.recall), pct(m.f1), m.support].forEach((v, i) => { const td = el("td", i ? "" : t, v); tr.append(td); }); $("#pertype tbody").append(tr);
  });
  R.test_predictions.forEach(p => {
    const d = el("div", "pred"), bad = p.gold.filter((g, i) => g !== p.pred[i]).length;
    d.append(el("small", "", `${p.id} · gold labels`), tokens(p.tokens, p.gold), el("small", "", `${p.id} · predicted${bad ? ` · ${bad} wrong (underlined)` : " · all correct"}`), tokens(p.tokens, p.pred, p.gold));
    $("#preds").append(d);
  });
  curve(R.training_curve);
}
function curve(log) {
  const svg = $("#curve"), W = 640, H = 220, L = 40, B = 30, ns = "http://www.w3.org/2000/svg";
  const add = (t, at, txt) => { const e = document.createElementNS(ns, t); for (const k in at) e.setAttribute(k, at[k]); if (txt) e.textContent = txt; svg.append(e); };
  const X = i => L + i / (log.length - 1) * (W - L - 16), Y = v => 14 + (1 - v) * (H - B - 14);
  [0, .5, 1].forEach(v => { add("line", { x1: L, x2: W - 16, y1: Y(v), y2: Y(v), stroke: "currentColor", opacity: .15 }); add("text", { x: 4, y: Y(v) + 4 }, v.toFixed(1)); });
  [["dev_f1", "var(--ORG)", "dev"], ["test_f1", "var(--PER)", "test"]].forEach(([k, col, n], j) => {
    add("polyline", { fill: "none", stroke: col, "stroke-width": 2.5, points: log.map((r, i) => `${X(i)},${Y(r[k])}`).join(" ") });
    add("text", { x: W - 90, y: 150 + j * 16, fill: col, style: `fill:${col}` }, `${n} F1`);
  });
  add("text", { x: L, y: H - 8 }, "epoch 1"); add("text", { x: W - 70, y: H - 8 }, "epoch " + log.length);
}

function showAsr(A) {
  const tb = $("#asrtab tbody"); let s = 0;
  A.rows.forEach(r => { const tr = el("tr", r.wer > 0 ? "bad" : ""); [r.file, r.ref, r.asr, r.wer.toFixed(3)].forEach(v => tr.append(el("td", "", v))); tb.append(tr); s += r.wer; });
  const e = A.end_to_end;
  $("#asrsum").textContent = `Mean word error rate ${(s / A.rows.length).toFixed(3)} over ${A.rows.length} recordings. Run end to end on the test recordings, the tagger recovered ${e.found} of ${e.total} reference entities exactly; most misses are mis-heard names.`;
}

/* ---------- boot ---------- */
(async function () {
  draw(); addEventListener("resize", () => !analyser && draw());
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { $("#mic").hidden = true; $("#status").textContent = "Voice input needs Chrome, Edge or Safari. You can still type any sentence."; }
  $("#mic").onclick = toggleMic; $("#inp").addEventListener("input", () => M && run());
  try {
    const [m, r, d] = await Promise.all(["model/bigru_ner.json", "data/results.json", "data/dataset.json"].map(u => fetch(u).then(x => x.json())));
    M = m; M.w2i = Object.fromEntries(m.words.map((w, i) => [w, i])); M.s2i = Object.fromEntries(m.suffixes.map((w, i) => [w, i])); R = r; DS = d;
    ["Sundar Pichai works at Google in Bengaluru", "narendra modi visited madurai and met officials from wipro", "Mahendra Singh Dhoni played for Chennai Super Kings", "Anita Desai joined Zoho after studying at IIT Madras"]
      .forEach(t => { const b = el("button", "", t); b.type = "button"; b.onclick = () => { $("#inp").value = t; run(); }; $("#presets").append(b); });
    run(); showDataset(); showResults();
    fetch("data/asr.json").then(x => x.json()).then(showAsr).catch(() => {});
  } catch (e) { $("#status").textContent = "Could not load the model files. Serve this folder over http (for example `npx serve website`) instead of opening index.html directly."; }
})();
