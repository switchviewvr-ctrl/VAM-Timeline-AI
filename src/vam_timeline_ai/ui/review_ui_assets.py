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
.meta div { min-width: 0; overflow-wrap: anywhere; }
.quick-review {
  display: grid;
  grid-template-columns: 1fr;
  gap: 8px;
  margin: 12px 0;
  padding: 10px;
  border: 1px solid #d7dce5;
  border-radius: 8px;
  background: #fbfcfe;
}
.quick-review textarea { min-height: 110px; }
.screenshot-drop {
  border: 1px dashed #9aa6b8;
  border-radius: 8px;
  padding: 10px;
  min-height: 74px;
  background: #fff;
  color: var(--muted);
  font-size: 13px;
}
.screenshot-drop:focus {
  outline: 2px solid #8ab4f8;
}
.screenshot-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.screenshot-list img {
  max-width: 120px;
  max-height: 90px;
  border: 1px solid #d7dce5;
  border-radius: 6px;
  object-fit: contain;
  background: #f8fafc;
}
.save-status {
  color: #2f6b45;
  font-size: 12px;
  min-height: 16px;
}
.system-proposal {
  margin: 10px 0;
  padding: 10px;
  border-left: 3px solid #97a6ba;
  background: #f8fafc;
  font-size: 13px;
}
.system-proposal .title { color: var(--muted); font-weight: 600; margin-bottom: 4px; }
.digital-twin-preview {
  margin: 10px 0;
  padding: 8px;
  border: 1px solid #d7dce5;
  border-radius: 8px;
  background: #fbfcfe;
}
.digital-twin-preview img {
  display: block;
  width: 100%;
  max-height: 520px;
  object-fit: contain;
  background: #fff;
  border: 1px solid #e5eaf2;
  border-radius: 6px;
}
.digital-twin-preview video {
  display: block;
  width: 100%;
  max-height: 520px;
  background: #fff;
  border: 1px solid #e5eaf2;
  border-radius: 6px;
}
.digital-twin-preview .title {
  color: var(--muted);
  font-size: 12px;
  font-weight: 600;
  margin-bottom: 6px;
}
.compact-facts {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4px 10px;
  font-size: 13px;
  margin: 8px 0;
}
.compact-facts b { color: var(--muted); font-weight: 500; }
.debug-details {
  margin: 8px 0;
  border: 1px solid #e5eaf2;
  border-radius: 6px;
  padding: 6px 8px;
  background: #fbfcfe;
}
.debug-details summary { cursor: pointer; color: var(--muted); font-size: 12px; }
.review-label {
  font-size: 17px;
  font-weight: 700;
  margin: 0 0 8px;
}
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
const FAMILIES = ["", "cowgirl", "bj_oral", "doggy", "standing_hand_head", "hand_gesture", "head_gesture", "receiver_response", "transition", "unknown"];
const REVIEW_LABELS = (DATA.answer_schema && DATA.answer_schema.review_labels) || [
  "correct_clean_cowgirl_motion", "correct_short_cowgirl_motion", "cowgirl_pose_only_low_motion",
  "cowgirl_transition_intro_alignment", "standing_hand_head_not_cowgirl", "bj_oral_not_cowgirl",
  "receiver_response_not_rider_motion", "wrong_partner_context", "wrong_contact_support",
  "broken_pose_or_bad_data", "unknown_unclear"
];
const ERROR_TAGS = (DATA.answer_schema && DATA.answer_schema.error_tags) || [
  "low_motion_hold", "intro_alignment", "bj_oral_as_cowgirl", "standing_hand_head_as_cowgirl",
  "receiver_as_rider", "contact_wrong_target", "partner_context_missing", "duplicate_review_selection",
  "foot_anchor_weird", "pose_broken", "controller_missing", "generation_safe_false_positive"
];
const REVIEW_QUESTIONS = (DATA.answer_schema && DATA.answer_schema.review_questions) || [];
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
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(answers));
    updateProgress();
    return true;
  } catch (err) {
    alert("Browser storage is full. Export answers, then clear local answers or use fewer screenshots. New pasted screenshots are compressed, but existing large screenshots may still fill storage.");
    return false;
  }
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
      review_labels: [],
      error_tags: [],
      screenshots: [],
      notes: ""
    };
  }
  return answers[id];
}

function setAnswer(id, key, value) {
  answerFor(id)[key] = value;
  const ok = saveAnswers();
  const card = document.querySelector(`[data-review-card="${id}"]`);
  if (card) card.classList.toggle("reviewed", isReviewed(answerFor(id)));
  return ok;
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
    , "torso_lean_direction", "facing_context", "hands_behind_support_score",
    "hands_on_partner_legs_score", "hands_on_partner_thighs_score",
    "partner_leg_support_confidence", "facing_confidence"
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
  card.appendChild(el("h2", {class:"review-label", text: item.review_label || item.semantic_review_label || id}));
  card.appendChild(el("div", {class:"pillrow"}, [
    el("span", {class:"pill " + familyClass(item.semantic_family), text:value(item.semantic_family)}),
    el("span", {class:"pill", text:value(item.why_selected || item.category)}),
    el("span", {class:"pill", text:"gen_safe: " + value(item.generation_safe)}),
  ]));
  if ((item.duplicate_status && item.duplicate_status !== "unique") || item.previously_reviewed) {
    card.appendChild(el("div", {class:"notice warn", text:item.review_trust_warning || "This item appears to overlap a previously reviewed sample/window."}));
  }
  card.appendChild(el("div", {class:"meta"}, [
    kv("Review ID", id),
    kv("Scene", item.source_scene_file),
    kv("Actor", item.technical_atom_id || item.technical_actor_id),
    kv("Clip", value(item.clip_name)),
    kv("Time", `${value(item.start_seconds)} - ${value(item.end_seconds)}s`),
    kv("Pose", `${value(item.pose_family || item.pose_semantics?.family)} / ${value(item.pose_subtype || item.pose_semantics?.subtype)}`),
    kv("Torso / Facing", `${value(item.torso_lean_direction)} / ${value(item.facing_context)}`),
    kv("Motion", `${value(item.motion_subtype || item.motion_semantics?.subtype)} / ${value(item.phase || item.motion_semantics?.phase)}`),
    kv("Clean gate", `${value(item.clean_motion_gate)} / ${value(item.clean_motion_gate_reason)}`),
    kv("Partner", item.partner_relation),
    kv("Contact", item.contact_support),
    kv("Support ctx", item.support_context),
    kv("Duplicate", `${value(item.duplicate_status)} / ${value(item.duplicate_group_id)}`),
    kv("Interaction", item.interaction_family),
    kv("Likely failure", item.likely_failure_mode),
  ]));
  card.appendChild(el("div", {class:"compact-facts"}, [
    kvInline("Pose", `${value(item.pose_subtype || item.pose_semantics?.subtype)}`),
    kvInline("Motion", `${value(item.motion_subtype || item.motion_semantics?.subtype)} / ${value(item.phase || item.motion_semantics?.phase)}`),
    kvInline("Torso", `${value(item.torso_lean_direction)} / ${value(item.facing_context)}`),
    kvInline("Support", `${value(item.contact_support)}`),
    kvInline("Time", `${value(item.start_seconds)}-${value(item.end_seconds)}s`),
    kvInline("Clip", value(item.clip_name)),
  ]));
  card.appendChild(el("div", {class:"system-proposal"}, [
    el("div", {class:"title", text:"system guess"}),
    el("div", {text:`${value(item.semantic_family)} / ${value(item.pose_subtype || item.pose_semantics?.subtype)} / ${value(item.motion_subtype || item.motion_semantics?.subtype)} / ${value(item.contact_support)}`}),
    el("div", {text:`bucket: ${value(item.why_selected || item.category)}`}),
  ]));
  if (item.digital_twin_gif || item.digital_twin_mp4 || item.digital_twin_contact_sheet_large || item.digital_twin_contact_sheet) {
    const media = [];
    if (item.digital_twin_gif) {
      media.push(el("a", {href:item.digital_twin_gif, target:"_blank"}, [
        el("img", {src:item.digital_twin_gif, alt:`${id} animated digital twin preview`})
      ]));
    } else if (item.digital_twin_mp4) {
      media.push(el("video", {controls:true, src:item.digital_twin_mp4}));
    } else if (item.digital_twin_contact_sheet_large) {
      media.push(el("a", {href:item.digital_twin_contact_sheet_large, target:"_blank"}, [
        el("img", {src:item.digital_twin_contact_sheet_large, alt:`${id} large digital twin contact sheet`})
      ]));
    } else if (item.digital_twin_contact_sheet) {
      media.push(el("div", {class:"notice warn", text:"Only static technical plot available; no animated digital-twin preview."}));
      media.push(el("a", {href:item.digital_twin_contact_sheet, target:"_blank"}, [
        el("img", {src:item.digital_twin_contact_sheet, alt:`${id} digital twin contact sheet`})
      ]));
    }
    const links = [];
    if (item.digital_twin_mp4) links.push(link(item.digital_twin_mp4, "MP4"));
    if (item.digital_twin_gif) links.push(link(item.digital_twin_gif, "GIF"));
    if (item.digital_twin_contact_sheet_large) links.push(link(item.digital_twin_contact_sheet_large, "large sheet"));
    if (item.digital_twin_contact_sheet) links.push(link(item.digital_twin_contact_sheet, "static plot"));
    card.appendChild(el("div", {class:"digital-twin-preview"}, [
      el("div", {class:"title", text:"digital twin preview"}),
      ...media,
      ...(links.length ? [el("div", {class:"pillrow"}, links)] : []),
      ...(item.digital_twin_primary_visual_type ? [el("div", {class:"pillrow"}, [
        el("span", {class:"pill", text:"primary: " + item.digital_twin_primary_visual_type}),
        el("span", {class:"pill", text:"quality: " + value(item.digital_twin_visual_quality)}),
      ])] : []),
      ...(item.digital_twin_warnings && item.digital_twin_warnings.length ? [el("div", {class:"warn", text:item.digital_twin_warnings.join("; ")})] : []),
    ]));
  }
  if (item.visual_judge || item.multisignal_priority) {
    card.appendChild(el("div", {class:"system-proposal"}, [
      el("div", {class:"title", text:"visual judge / multisignal"}),
      el("div", {class:"compact-facts"}, [
        kvInline("VLM family", `${value(item.visual_suggested_family)} (${value(item.visual_family_confidence)})`),
        kvInline("Parse", value(item.visual_parse_status)),
        kvInline("Pose", `${value(item.visual_body_pose_guess)} / torso ${value(item.visual_torso_lean_guess)}`),
        kvInline("Facing", value(item.visual_facing_guess)),
        kvInline("Partner", `visible=${value(item.visual_partner_visible)}`),
        kvInline("Motion", `${value(item.visual_motion_visible)} / ${value(item.visual_dominant_motion_guess)}`),
        kvInline("Contact", value(item.visual_contact_support_guess)),
        kvInline("Priority", `${value(item.multisignal_priority)} / ${value(item.multisignal_reason)}`),
      ]),
      ...(item.visual_reasoning_short ? [el("div", {text:"reasoning: " + item.visual_reasoning_short})] : []),
      el("div", {class:"pillrow"}, [
        buttonSmall("VLM correct", () => setAnswer(id, "visual_judge_verdict", "correct")),
        buttonSmall("VLM wrong", () => setAnswer(id, "visual_judge_verdict", "wrong")),
        buttonSmall("VLM unsure", () => setAnswer(id, "visual_judge_verdict", "unsure")),
      ]),
    ]));
  }
  if (item.ontology_resolved_family || item.ontology_match) {
    card.appendChild(el("div", {class:"system-proposal"}, [
      el("div", {class:"title", text:"ontology / pose-first semantics"}),
      el("div", {class:"compact-facts"}, [
        kvInline("Ontology family", value(item.ontology_resolved_family)),
        kvInline("Motion", value(item.ontology_resolved_motion_subtype)),
        kvInline("Primary driver", value(item.ontology_primary_motion_center)),
        kvInline("Target", value(item.ontology_target_region)),
        kvInline("Clean gate", value(item.ontology_clean_motion_gate)),
        kvInline("Match", value(item.ontology_match)),
        kvInline("Review priority", value(item.ontology_review_priority)),
      ]),
      ...(item.ontology_conflict_flags && item.ontology_conflict_flags.length ? [el("div", {class:"warn", text:"conflicts: " + item.ontology_conflict_flags.join("; ")})] : []),
      ...(item.ontology_conflicts && item.ontology_conflicts.length ? [el("div", {class:"warn", text:"alignment conflicts: " + item.ontology_conflicts.join("; ")})] : []),
      ...(item.ontology_missing_requirements && item.ontology_missing_requirements.length ? [el("div", {class:"warn", text:"missing: " + item.ontology_missing_requirements.join("; ")})] : []),
      ...(item.ontology_not_labels && item.ontology_not_labels.length ? [el("div", {class:"pillrow"}, item.ontology_not_labels.map(x => el("span", {class:"pill", text:"not: " + x})))] : []),
      ...(item.ontology_explanation ? [el("div", {text:item.ontology_explanation})] : []),
    ]));
  }
  card.appendChild(quickAnswerForm(id, a));
  const details = el("details", {class:"debug-details"}, [
    el("summary", {text:"source/debug details"}),
    el("div", {class:"meta"}, [
      kv("Scene", item.source_scene_file),
      kv("Actor", item.technical_atom_id || item.technical_actor_id),
      kv("Partner", item.partner_relation),
      kv("Clean gate", `${value(item.clean_motion_gate)} / ${value(item.clean_motion_gate_reason)}`),
      kv("Likely failure", item.likely_failure_mode),
      kv("Duplicate status", `${value(item.duplicate_status)} / overlaps: ${value(item.overlaps_with_review_ids)}`),
      kv("Full scene path", item.source_scene_path || item.source_scene_file),
      kv("Source ID", item.source_id),
      kv("Timeline / clip", `${value(item.storable_id || item.plugin_id)} / ${value(item.clip_name)} #${value(item.clip_index)}`),
      kv("Window", item.window_id),
    ])
  ]);
  card.appendChild(details);
  const evidence = el("details", {class:"debug-details"}, [
    el("summary", {text:"evidence scores"}),
    ...(item.motion_metrics ? [el("pre", {text:"Motion metrics\n" + JSON.stringify(item.motion_metrics, null, 2)})] : []),
    scoreBlock(item),
    ...(item.warnings && item.warnings.length ? [el("p", {class:"warn", text:"Warnings: " + item.warnings.join("; ")})] : []),
  ]);
  card.appendChild(evidence);
  const links = [];
  if (item.item_review_path) links.push(link(item.item_review_path, "item instructions"));
  if (item.timeline_export_path) links.push(link(item.timeline_export_path, "timeline segment"));
  if (item.vam_animation_path) links.push(link(item.vam_animation_path, "VaM animations copy"));
  if (item.source_scene_path) links.push(el("span", {class:"pill", text:item.source_scene_path}));
  if (links.length) card.appendChild(el("div", {class:"pillrow"}, links));
  return card;
}

function kvInline(k, v) {
  return el("div", {}, [el("b", {text:k + ": "}), value(v)]);
}

function kv(k, v) {
  return el("div", {}, [el("div", {class:"k", text:k}), el("div", {text:value(v)})]);
}

function link(href, text) {
  return el("a", {class:"button-link", href:href, target:"_blank", text:text});
}

function buttonSmall(text, onClick) {
  const b = el("button", {type:"button", text:text});
  b.addEventListener("click", onClick);
  return b;
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

function quickAnswerForm(id, a) {
  const wrap = el("div", {class:"answer"});
  wrap.className = "quick-review";
  const notes = el("textarea", {placeholder:"Schreib einfach rein, was du siehst. Beispiel: 'Pose ist normal cowgirl, Animation ist auch normal cowgirl.'"});
  notes.value = a.notes || "";
  notes.addEventListener("input", () => setAnswer(id, "notes", notes.value));
  wrap.appendChild(notes);
  wrap.appendChild(screenshotBox(id));
  const status = el("div", {class:"save-status", text:""});
  const save = el("button", {type:"button", text:"Save answer"});
  save.addEventListener("click", () => {
    saveAnswers();
    status.textContent = "saved locally";
    setTimeout(() => { status.textContent = ""; }, 1600);
  });
  wrap.appendChild(el("div", {class:"pillrow"}, [save, status]));
  return wrap;
}

function screenshotBox(id) {
  const box = el("div", {class:"screenshot-drop", tabindex:"0", text:"Screenshot hier anklicken und mit Ctrl+V einfuegen. Bilder werden verkleinert gespeichert."});
  const list = el("div", {class:"screenshot-list"});
  const render = () => {
    list.innerHTML = "";
    const shots = answerFor(id).screenshots || [];
    shots.forEach((shot, idx) => {
      const img = el("img", {src:shot.data_url || shot, alt:`screenshot ${idx + 1}`});
      const remove = el("button", {type:"button", text:"remove"});
      remove.addEventListener("click", () => {
        const cur = answerFor(id).screenshots || [];
        cur.splice(idx, 1);
        setAnswer(id, "screenshots", cur);
        render();
      });
      list.appendChild(el("div", {}, [img, remove]));
    });
  };
  box.addEventListener("paste", async (event) => {
    const items = Array.from(event.clipboardData?.items || []);
    for (const item of items) {
      if (!item.type.startsWith("image/")) continue;
      const file = item.getAsFile();
      if (!file) continue;
      const cur = answerFor(id).screenshots || [];
      const previous = cur.slice();
      const dataUrl = await readCompressedImageAsDataURL(file);
      cur.push({
        name:`screenshot_${cur.length + 1}.jpg`,
        mime:"image/jpeg",
        original_mime:file.type,
        compressed:true,
        data_url:dataUrl,
        captured_at:new Date().toISOString()
      });
      answerFor(id).screenshots = cur;
      if (!saveAnswers()) {
        answerFor(id).screenshots = previous;
        saveAnswers();
        box.textContent = "Speicher voll. Answers exportieren oder alte Screenshots entfernen.";
        return;
      }
      const kb = Math.round(dataUrl.length * 0.75 / 1024);
      box.textContent = `Screenshot gespeichert (~${kb} KB). Weitere mit Ctrl+V einfuegen.`;
      render();
      event.preventDefault();
    }
  });
  render();
  return el("div", {}, [box, list]);
}

function readCompressedImageAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        const maxDim = 900;
        const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(img.width * scale));
        canvas.height = Math.max(1, Math.round(img.height * scale));
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL("image/jpeg", 0.72));
      };
      img.onerror = reject;
      img.src = reader.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
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
  for (const r of rows) lines.push(`${r.review_id}: verdict=${r.verdict || "unknown"} labels=${(r.review_labels||[]).join(", ")} family=${r.actual_semantic_family || "unknown"} pose=${r.actual_pose || ""} motion=${r.actual_motion || ""} contact=${r.actual_contact_support || ""} errors=${(r.error_tags||[]).join(", ")} notes=${r.notes || ""}`);
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
