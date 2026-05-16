"""Embedded assets for the local semantic review workbench."""

from __future__ import annotations


STYLE_CSS = r"""
:root {
  color-scheme: light;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --ink: #172033;
  --muted: #667085;
  --line: #d7dce5;
  --cowgirl: #1f8a4c;
  --bj: #246bcb;
  --receiver: #b45b16;
  --unknown: #667085;
  --warn: #b42318;
  --soft: #fff7df;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", Arial, sans-serif;
  color: var(--ink);
  background: var(--bg);
}
header {
  position: sticky;
  top: 0;
  z-index: 10;
  padding: 14px 20px;
  background: #101828;
  color: white;
  border-bottom: 1px solid #0b1220;
}
header h1 { margin: 0 0 6px; font-size: 20px; }
header .sub { color: #cbd5e1; font-size: 13px; }
nav {
  display: flex;
  gap: 8px;
  padding: 10px 20px;
  background: #e9edf5;
  border-bottom: 1px solid var(--line);
  position: sticky;
  top: 72px;
  z-index: 9;
  flex-wrap: wrap;
}
button, select, input, textarea {
  font: inherit;
}
button, .button-link {
  border: 1px solid #b9c0cd;
  background: #fff;
  color: #172033;
  border-radius: 6px;
  padding: 7px 10px;
  cursor: pointer;
  text-decoration: none;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
button.active { background: #172033; color: white; border-color: #172033; }
button.primary { background: #1b4ed8; color: white; border-color: #1b4ed8; }
button.danger { color: #b42318; border-color: #f0b8b1; }
main { padding: 18px 20px 40px; }
.page { display: none; }
.page.active { display: block; }
.toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: end;
  margin-bottom: 14px;
  padding: 12px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
}
.field { display: flex; flex-direction: column; gap: 4px; min-width: 160px; }
label { font-size: 12px; color: var(--muted); }
select, input[type="text"], input[type="number"], textarea {
  border: 1px solid #c9cfda;
  border-radius: 6px;
  padding: 7px 8px;
  background: #fff;
  color: var(--ink);
}
textarea { width: 100%; min-height: 78px; resize: vertical; }
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  gap: 14px;
}
.card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
  box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}
.card.reviewed { border-color: #3aa76d; box-shadow: 0 0 0 2px rgba(58, 167, 109, 0.14); }
.card h2, .card h3 { margin: 0 0 8px; font-size: 16px; }
.meta {
  display: grid;
  grid-template-columns: 120px 1fr;
  gap: 4px 10px;
  font-size: 13px;
}
.meta .k { color: var(--muted); }
.pillrow { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
.pill {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 3px 8px;
  font-size: 12px;
  background: #eef2f7;
  border: 1px solid #dde3ee;
}
.family-cowgirl { background: #e7f6ed; color: var(--cowgirl); border-color: #c2e8d1; }
.family-bj_oral { background: #e7f0ff; color: var(--bj); border-color: #c7dcff; }
.family-receiver_response { background: #fff0e0; color: var(--receiver); border-color: #ffd6a8; }
.family-unknown { background: #eef2f7; color: var(--unknown); }
.warn { color: var(--warn); }
.scores {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 5px 10px;
  font-size: 12px;
  background: #f8fafc;
  border: 1px solid #e5eaf2;
  padding: 8px;
  border-radius: 6px;
}
.answer {
  margin-top: 10px;
  border-top: 1px solid var(--line);
  padding-top: 10px;
  display: grid;
  gap: 8px;
}
.answer-row {
  display: grid;
  grid-template-columns: minmax(130px, 0.8fr) minmax(150px, 1fr);
  gap: 8px;
}
.tags {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
  gap: 5px;
  padding: 8px;
  background: #fafafa;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
}
.tags label { color: var(--ink); font-size: 12px; }
table {
  width: 100%;
  border-collapse: collapse;
  background: var(--panel);
  border: 1px solid var(--line);
}
th, td {
  border-bottom: 1px solid #e7ebf2;
  padding: 7px 8px;
  text-align: left;
  font-size: 12px;
  vertical-align: top;
}
th { background: #f1f4f9; position: sticky; top: 122px; z-index: 2; }
.notice {
  padding: 12px;
  background: var(--soft);
  border: 1px solid #f4d783;
  border-radius: 8px;
  margin-bottom: 14px;
}
.split {
  display: grid;
  grid-template-columns: minmax(300px, 1fr) minmax(300px, 1fr);
  gap: 14px;
}
.progress {
  height: 8px;
  background: #e5eaf2;
  border-radius: 999px;
  overflow: hidden;
  min-width: 180px;
}
.progress span { display: block; height: 100%; background: #1f8a4c; width: 0%; }
pre {
  white-space: pre-wrap;
  background: #0f172a;
  color: #e2e8f0;
  padding: 12px;
  border-radius: 8px;
  overflow: auto;
}
@media (max-width: 760px) {
  nav { top: 92px; }
  .grid, .split { grid-template-columns: 1fr; }
  .answer-row, .meta { grid-template-columns: 1fr; }
}
"""


APP_JS = r"""
const DATA = window.REVIEW_UI_DATA || {};
const STORE_KEY = "vam_timeline_ai_review_answers_" + (DATA.review_name || "review");
const YES_NO = ["unknown", "true", "false", "not_applicable"];
const FAMILIES = ["", "cowgirl", "bj_oral", "doggy", "standing_hand_head", "hand_gesture", "head_gesture", "receiver_response", "transition", "unknown"];
const VERDICTS = ["", "correct", "partially_correct", "wrong", "unclear", "unavailable"];
const ERROR_TAGS = [
  "low_motion_hold", "intro_alignment", "bj_oral_as_cowgirl", "standing_hand_head_as_cowgirl",
  "receiver_as_rider", "contact_wrong_target", "partner_context_missing", "duplicate_review_selection",
  "foot_anchor_weird", "pose_broken", "controller_missing", "generation_safe_false_positive"
];
let answers = loadAnswers();
let currentReviewFilter = "";

function el(tag, attrs={}, children=[]) {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v == null ? "" : String(v);
    else if (k === "html") node.innerHTML = v;
    else node.setAttribute(k, v);
  });
  for (const child of children) node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  return node;
}

function loadAnswers() {
  try { return JSON.parse(localStorage.getItem(STORE_KEY) || "{}"); }
  catch { return {}; }
}

function saveAnswers() {
  localStorage.setItem(STORE_KEY, JSON.stringify(answers));
  updateProgress();
}

function answerFor(id) {
  if (!answers[id]) {
    answers[id] = {
      review_id: id,
      semantic_family_correct: "unknown",
      actual_semantic_family: "",
      pose_correct: "unknown",
      actual_pose: "",
      motion_correct: "unknown",
      actual_motion: "",
      partner_relation_correct: "unknown",
      actual_partner_relation: "",
      contact_support_correct: "unknown",
      actual_contact_support: "",
      generation_safe_correct: "unknown",
      actual_generation_safe: "unknown",
      verdict: "",
      error_tags: [],
      notes: ""
    };
  }
  return answers[id];
}

function setAnswer(id, key, value) {
  answerFor(id)[key] = value;
  saveAnswers();
  const card = document.querySelector(`[data-review-card="${id}"]`);
  if (card) card.classList.toggle("reviewed", isReviewed(answerFor(id)));
}

function isReviewed(a) {
  return Boolean(a && (a.verdict || a.notes || (a.error_tags && a.error_tags.length) || FAMILIES.includes(a.actual_semantic_family) && a.actual_semantic_family));
}

function tab(name) {
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  document.querySelectorAll("nav button").forEach(b => b.classList.remove("active"));
  document.getElementById("page-" + name).classList.add("active");
  document.getElementById("tab-" + name).classList.add("active");
}

function familyClass(f) {
  if (!f) return "family-unknown";
  if (f === "cowgirl") return "family-cowgirl";
  if (f === "bj_oral") return "family-bj_oral";
  if (f === "receiver_response") return "family-receiver_response";
  return "family-unknown";
}

function value(v) {
  if (Array.isArray(v)) return v.join(", ");
  if (v === true) return "true";
  if (v === false) return "false";
  if (v == null || v === "") return "-";
  return String(v);
}

function fmtNum(v) {
  if (v == null || v === "") return "-";
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(3) : value(v);
}

function scoreBlock(item) {
  const keys = [
    "pose_confidence", "motion_score", "interaction_score", "hands_on_partner_chest_score",
    "hands_on_partner_hips_score", "partner_lying_score", "pelvis_alignment_score",
    "rider_above_partner_score", "contact_support_confidence", "clean_motion_gate",
    "hip_motion_strength", "pelvis_trajectory_strength", "pelvis_cycle_count",
    "motion_duration_confidence"
  ];
  return el("div", {class:"scores"}, keys.map(k => {
    const raw = item[k] ?? (item.evidence_scores || {})[k];
    const shown = k === "clean_motion_gate" ? value(raw) : fmtNum(raw);
    return el("div", {}, [el("b", {text:k + ": "}), shown]);
  }));
}

function renderReviewBatch() {
  const root = document.getElementById("review-list");
  root.innerHTML = "";
  const items = DATA.review_items || [];
  const filtered = currentReviewFilter ? items.filter(i => JSON.stringify(i).toLowerCase().includes(currentReviewFilter.toLowerCase())) : items;
  document.getElementById("review-count").textContent = `${filtered.length} / ${items.length} shown`;
  for (const item of filtered) root.appendChild(reviewCard(item));
  updateProgress();
}

function reviewCard(item) {
  const id = item.review_id || item.window_id;
  const a = answerFor(id);
  const card = el("section", {class:"card" + (isReviewed(a) ? " reviewed" : ""), "data-review-card": id});
  card.appendChild(el("h2", {text: id}));
  card.appendChild(el("div", {class:"pillrow"}, [
    el("span", {class:"pill " + familyClass(item.semantic_family), text:value(item.semantic_family)}),
    el("span", {class:"pill", text:value(item.why_selected || item.category)}),
    el("span", {class:"pill", text:"gen_safe: " + value(item.generation_safe)}),
  ]));
  card.appendChild(el("div", {class:"meta"}, [
    kv("Scene", item.source_scene_path || item.source_scene_file),
    kv("Actor", item.technical_atom_id || item.technical_actor_id),
    kv("Time", `${value(item.start_seconds)} - ${value(item.end_seconds)}s`),
    kv("Pose", `${value(item.pose_family || item.pose_semantics?.family)} / ${value(item.pose_subtype || item.pose_semantics?.subtype)}`),
    kv("Motion", `${value(item.motion_subtype || item.motion_semantics?.subtype)} / ${value(item.phase || item.motion_semantics?.phase)}`),
    kv("Clean gate", `${value(item.clean_motion_gate)} / ${value(item.clean_motion_gate_reason)}`),
    kv("Partner", item.partner_relation),
    kv("Contact", item.contact_support),
    kv("Interaction", item.interaction_family),
  ]));
  card.appendChild(scoreBlock(item));
  if (item.warnings && item.warnings.length) card.appendChild(el("p", {class:"warn", text:"Warnings: " + item.warnings.join("; ")}));
  const links = [];
  if (item.item_review_path) links.push(link(item.item_review_path, "item instructions"));
  if (item.timeline_export_path) links.push(link(item.timeline_export_path, "timeline segment"));
  if (item.source_scene_path) links.push(el("span", {class:"pill", text:item.source_scene_path}));
  if (links.length) card.appendChild(el("div", {class:"pillrow"}, links));
  card.appendChild(answerForm(id, a));
  return card;
}

function kv(k, v) {
  return el("div", {}, [el("div", {class:"k", text:k}), el("div", {text:value(v)})]);
}

function link(href, text) {
  return el("a", {class:"button-link", href:href, target:"_blank", text:text});
}

function selectField(id, key, options) {
  const a = answerFor(id);
  const s = el("select");
  for (const opt of options) s.appendChild(el("option", {value:opt, text:opt || "-"}));
  s.value = a[key] || "";
  s.addEventListener("change", () => setAnswer(id, key, s.value));
  return s;
}

function inputField(id, key, placeholder="") {
  const a = answerFor(id);
  const input = el("input", {type:"text", placeholder});
  input.value = a[key] || "";
  input.addEventListener("input", () => setAnswer(id, key, input.value));
  return input;
}

function answerForm(id, a) {
  const wrap = el("div", {class:"answer"});
  const rows = [
    ["semantic_family_correct", selectField(id, "semantic_family_correct", YES_NO)],
    ["actual_semantic_family", selectField(id, "actual_semantic_family", FAMILIES)],
    ["pose_correct", selectField(id, "pose_correct", YES_NO)],
    ["actual_pose", inputField(id, "actual_pose", "e.g. cowgirl_lean_forward_supported")],
    ["motion_correct", selectField(id, "motion_correct", YES_NO)],
    ["actual_motion", inputField(id, "actual_motion", "e.g. clean grinding / intro alignment")],
    ["partner_relation_correct", selectField(id, "partner_relation_correct", YES_NO)],
    ["actual_partner_relation", inputField(id, "actual_partner_relation", "e.g. rider over receiver")],
    ["contact_support_correct", selectField(id, "contact_support_correct", YES_NO)],
    ["actual_contact_support", inputField(id, "actual_contact_support", "e.g. hands_on_partner_chest")],
    ["generation_safe_correct", selectField(id, "generation_safe_correct", YES_NO)],
    ["actual_generation_safe", selectField(id, "actual_generation_safe", YES_NO)],
    ["verdict", selectField(id, "verdict", VERDICTS)],
  ];
  for (const [label, field] of rows) wrap.appendChild(el("div", {class:"answer-row"}, [el("label", {text:label}), field]));
  const tags = el("div", {class:"tags"});
  for (const tag of ERROR_TAGS) {
    const cb = el("input", {type:"checkbox"});
    cb.checked = (a.error_tags || []).includes(tag);
    cb.addEventListener("change", () => {
      const cur = new Set(answerFor(id).error_tags || []);
      cb.checked ? cur.add(tag) : cur.delete(tag);
      setAnswer(id, "error_tags", Array.from(cur));
    });
    tags.appendChild(el("label", {}, [cb, " " + tag]));
  }
  wrap.appendChild(el("label", {text:"error tags"}));
  wrap.appendChild(tags);
  const notes = el("textarea", {placeholder:"notes"});
  notes.value = a.notes || "";
  notes.addEventListener("input", () => setAnswer(id, "notes", notes.value));
  wrap.appendChild(el("label", {text:"notes"}));
  wrap.appendChild(notes);
  return wrap;
}

function updateProgress() {
  const total = (DATA.review_items || []).length;
  const reviewed = (DATA.review_items || []).filter(i => isReviewed(answers[i.review_id || i.window_id])).length;
  document.getElementById("progress-text").textContent = `${reviewed} / ${total} reviewed`;
  document.getElementById("progress-bar").style.width = total ? `${reviewed * 100 / total}%` : "0%";
}

function renderExplorer() {
  const rows = DATA.candidates || [];
  const filters = ["semantic_family", "category", "pose_subtype", "motion_subtype", "phase", "partner_relation", "contact_support", "generation_safe", "source_scene_file"];
  const bar = document.getElementById("candidate-filters");
  bar.innerHTML = "";
  for (const f of filters) {
    const values = Array.from(new Set(rows.map(r => value(r[f])).filter(v => v && v !== "-"))).sort().slice(0, 200);
    const select = el("select", {"data-filter":f}, [el("option", {value:"", text:f + ": any"})]);
    values.forEach(v => select.appendChild(el("option", {value:v, text:v})));
    select.addEventListener("change", renderCandidateTable);
    bar.appendChild(el("div", {class:"field"}, [el("label", {text:f}), select]));
  }
  const presets = [
    "cowgirl_clean_motion_generation_safe", "cowgirl_pose_context_low_motion", "cowgirl_intro_alignment",
    "cowgirl_hands_on_partner_chest", "ambiguous_partner_contact", "bj_oral", "receiver_response",
    "standing_hand_head", "unknown_or_unusable"
  ];
  const preset = el("select", {}, [el("option", {value:"", text:"preset"})]);
  presets.forEach(p => preset.appendChild(el("option", {value:p, text:p})));
  preset.addEventListener("change", () => { document.getElementById("candidate-search").value = preset.value; renderCandidateTable(); });
  bar.appendChild(el("div", {class:"field"}, [el("label", {text:"preset search"}), preset]));
  renderCandidateTable();
}

function renderCandidateTable() {
  const rows = DATA.candidates || [];
  const search = (document.getElementById("candidate-search").value || "").toLowerCase();
  const active = {};
  document.querySelectorAll("#candidate-filters select[data-filter]").forEach(s => { if (s.value) active[s.getAttribute("data-filter")] = s.value; });
  let filtered = rows.filter(r => {
    if (search && !JSON.stringify(r).toLowerCase().includes(search)) return false;
    for (const [k, v] of Object.entries(active)) if (value(r[k]) !== v) return false;
    return true;
  });
  document.getElementById("candidate-summary").textContent = `${filtered.length} / ${rows.length} candidates`;
  const counts = countBy(filtered, "category");
  document.getElementById("candidate-counts").innerHTML = Object.entries(counts).slice(0, 12).map(([k,v]) => `<span class="pill">${k}: ${v}</span>`).join(" ");
  const tbody = document.getElementById("candidate-table-body");
  tbody.innerHTML = "";
  for (const r of filtered.slice(0, 500)) {
    tbody.appendChild(el("tr", {}, ["window_id","semantic_family","category","pose_subtype","motion_subtype","phase","contact_support","generation_safe","source_scene_file"].map(k => el("td", {text:value(r[k])}))));
  }
}

function countBy(rows, key) {
  const out = {};
  rows.forEach(r => { const v = value(r[key]); out[v] = (out[v] || 0) + 1; });
  return Object.fromEntries(Object.entries(out).sort((a,b) => b[1]-a[1]));
}

function renderHypotheses() {
  const root = document.getElementById("hypotheses");
  root.innerHTML = "";
  for (const h of DATA.hypotheses || []) {
    const card = el("section", {class:"card"}, [
      el("h3", {text:h.name}),
      el("p", {text:h.description}),
      el("div", {class:"pillrow"}, [
        el("span", {class:"pill", text:"count: " + h.count}),
        el("span", {class:"pill", text:"avg confidence: " + fmtNum(h.average_confidence)}),
        el("span", {class:"pill", text:"priority: " + h.review_priority}),
      ]),
      el("p", {text:"Recommended examples: " + (h.recommended_examples || []).join(", ")}),
    ]);
    root.appendChild(card);
  }
}

function renderErrors() {
  const root = document.getElementById("error-review");
  root.innerHTML = "";
  const taxonomy = DATA.error_taxonomy || {};
  if (!taxonomy.available) {
    root.appendChild(el("div", {class:"notice", text:"No human review ledger yet. Export answers, then run ingest-review-ui-answers and build-human-review-ledger."}));
    return;
  }
  root.appendChild(el("h2", {text:"Error taxonomy counts"}));
  const table = el("table", {}, [el("thead", {}, [el("tr", {}, [el("th", {text:"error"}), el("th", {text:"count"}), el("th", {text:"examples"})])]), el("tbody")]);
  const tbody = table.querySelector("tbody");
  for (const row of taxonomy.rows || []) tbody.appendChild(el("tr", {}, [el("td", {text:row.error}), el("td", {text:row.count}), el("td", {text:(row.examples || []).join(", ")})]));
  root.appendChild(table);
}

function renderStatus() {
  const root = document.getElementById("status-content");
  root.innerHTML = "";
  root.appendChild(el("pre", {text: JSON.stringify(DATA.status || {}, null, 2)}));
}

function exportJsonl() {
  const rows = Object.values(answers).filter(isReviewed);
  download("human_review_ui_answers.jsonl", rows.map(r => JSON.stringify(r)).join("\n") + "\n", "application/jsonl");
}

function yamlScalar(v) {
  if (Array.isArray(v)) return "\n" + v.map(x => `      - ${String(x).replace(/:/g, "\\:")}`).join("\n");
  if (v === true || v === false) return String(v);
  return JSON.stringify(v == null ? "" : v);
}

function exportYaml() {
  const rows = Object.values(answers).filter(isReviewed);
  let text = "reviews:\n";
  for (const r of rows) {
    text += `  ${r.review_id}:\n`;
    for (const [k, v] of Object.entries(r)) {
      if (k === "review_id") continue;
      text += `    ${k}: ${yamlScalar(v)}\n`;
    }
  }
  download("human_review_ui_answers.yaml", text, "text/yaml");
}

function codexPrompt() {
  const rows = Object.values(answers).filter(isReviewed);
  const lines = ["We reviewed " + (DATA.review_name || "the review") + ". Findings:"];
  for (const r of rows) lines.push(`${r.review_id}: verdict=${r.verdict || "unknown"} family=${r.actual_semantic_family || "unknown"} pose=${r.actual_pose || ""} motion=${r.actual_motion || ""} contact=${r.actual_contact_support || ""} errors=${(r.error_tags||[]).join(", ")} notes=${r.notes || ""}`);
  lines.push("Common errors: " + Array.from(new Set(rows.flatMap(r => r.error_tags || []))).join(", "));
  lines.push("Next needed fixes: calibrate the recurring errors above; do not treat these audit answers as manual_labels.yaml.");
  document.getElementById("codex-prompt").value = lines.join("\n");
}

async function saveServer() {
  const rows = Object.values(answers).filter(isReviewed);
  try {
    const res = await fetch("/api/save-answers", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({answers: rows})});
    const text = await res.text();
    alert(text);
  } catch (err) {
    alert("Server save unavailable in static mode. Use Download JSONL/YAML instead.");
  }
}

function download(name, text, type) {
  const blob = new Blob([text], {type});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

function init() {
  document.querySelectorAll("nav button").forEach(b => b.addEventListener("click", () => tab(b.dataset.tab)));
  document.getElementById("review-search").addEventListener("input", e => { currentReviewFilter = e.target.value; renderReviewBatch(); });
  document.getElementById("candidate-search").addEventListener("input", renderCandidateTable);
  document.getElementById("download-jsonl").addEventListener("click", exportJsonl);
  document.getElementById("download-yaml").addEventListener("click", exportYaml);
  document.getElementById("make-prompt").addEventListener("click", codexPrompt);
  document.getElementById("server-save").addEventListener("click", saveServer);
  document.getElementById("clear-local").addEventListener("click", () => { if (confirm("Clear local answers?")) { answers = {}; localStorage.removeItem(STORE_KEY); renderReviewBatch(); }});
  renderReviewBatch();
  renderExplorer();
  renderHypotheses();
  renderErrors();
  renderStatus();
  tab("review");
}

document.addEventListener("keydown", ev => {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
  if (ev.key === "c" || ev.key === "w" || ev.key === "u") {
    const first = (DATA.review_items || [])[0];
    if (!first) return;
    const id = first.review_id || first.window_id;
    setAnswer(id, "verdict", ev.key === "c" ? "correct" : ev.key === "w" ? "wrong" : "unclear");
    renderReviewBatch();
  }
});
document.addEventListener("DOMContentLoaded", init);
"""


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VaM Timeline AI Semantic Review Workbench</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <header>
    <h1>Semantic Review Workbench</h1>
    <div class="sub">Audit-only annotation UI. Does not modify manual_labels.yaml.</div>
  </header>
  <nav>
    <button id="tab-review" data-tab="review">Review Batch</button>
    <button id="tab-candidates" data-tab="candidates">Candidate DB Explorer</button>
    <button id="tab-hypotheses" data-tab="hypotheses">Hypothesis Tester</button>
    <button id="tab-errors" data-tab="errors">Error Review</button>
    <button id="tab-export" data-tab="export">Export Answers</button>
    <button id="tab-status" data-tab="status">Status / Reports</button>
    <div class="progress"><span id="progress-bar"></span></div>
    <span id="progress-text"></span>
  </nav>
  <main>
    <section id="page-review" class="page">
      <div class="toolbar">
        <div class="field"><label>Search review cards</label><input id="review-search" type="text" placeholder="scene, family, tag..."></div>
        <div><strong id="review-count"></strong></div>
        <button id="clear-local" class="danger">Clear Local Answers</button>
      </div>
      <div id="review-list" class="grid"></div>
    </section>
    <section id="page-candidates" class="page">
      <div class="notice">Explorer uses a compact local projection of the candidate DB. It is for triage, not ground truth.</div>
      <div class="toolbar">
        <div class="field"><label>Search</label><input id="candidate-search" type="text" placeholder="category, scene, warning..."></div>
        <div id="candidate-filters" class="toolbar"></div>
      </div>
      <div class="pillrow" id="candidate-counts"></div>
      <p id="candidate-summary"></p>
      <table>
        <thead><tr><th>window</th><th>family</th><th>category</th><th>pose</th><th>motion</th><th>phase</th><th>contact</th><th>safe</th><th>scene</th></tr></thead>
        <tbody id="candidate-table-body"></tbody>
      </table>
    </section>
    <section id="page-hypotheses" class="page">
      <div id="hypotheses" class="grid"></div>
    </section>
    <section id="page-errors" class="page">
      <div id="error-review"></div>
    </section>
    <section id="page-export" class="page">
      <div class="split">
        <section class="card">
          <h2>Export Answers</h2>
          <p>Answers are saved immediately in browser localStorage. Download them and keep them in the review folder when done.</p>
          <div class="pillrow">
            <button id="download-jsonl" class="primary">Download answers JSONL</button>
            <button id="download-yaml">Download answers YAML</button>
            <button id="server-save">Save via local server</button>
          </div>
        </section>
        <section class="card">
          <h2>Codex Summary Prompt</h2>
          <button id="make-prompt">Generate Prompt Text</button>
          <textarea id="codex-prompt" style="min-height:260px"></textarea>
        </section>
      </div>
    </section>
    <section id="page-status" class="page">
      <div id="status-content"></div>
    </section>
  </main>
  <script src="review_data.js"></script>
  <script src="app.js"></script>
</body>
</html>
"""
