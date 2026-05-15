from pathlib import Path

import numpy as np
import yaml

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.ml.dataset_v4 import build_ml_dataset_v4
from vam_timeline_ai.ml.silver_baseline_v1 import train_silver_baseline_v1
from vam_timeline_ai.ml.silver_readiness_v2 import analyze_silver_readiness_v2
from vam_timeline_ai.semantics.machine_label_aggregation import aggregate_machine_labels_v2
from vam_timeline_ai.semantics.machine_label_audit import audit_machine_labels_v1
from vam_timeline_ai.semantics.machine_proposal_review_batch_v2 import build_machine_proposal_review_batch_v2
from vam_timeline_ai.semantics.silver_labels_v2 import build_silver_labels_v2


def _proposal(window_id, label, confidence=0.9, ptype="positive", pair_window_id=None, rule="r1", source="machine_rule_v1"):
    return {
        "proposal_id": f"p_{window_id}_{label}_{pair_window_id}_{rule}",
        "window_id": window_id,
        "pair_window_id": pair_window_id,
        "sample_id": f"sample_{window_id}",
        "source_id": f"src_{window_id}",
        "source_scene_file": f"scene_{window_id[-1]}.json",
        "technical_atom_id": "Actor",
        "label": label,
        "label_group": "movement",
        "proposal_type": ptype,
        "confidence": confidence,
        "source": source,
        "rule_id": rule,
        "evidence_features": ["pelvis_movement_energy"],
        "evidence_values": {"pelvis_movement_energy": 1.0},
        "warnings": [],
        "is_silver_candidate": confidence >= 0.75 and ptype != "uncertain",
        "is_human_ground_truth": False,
    }


def test_machine_label_audit_detects_duplicates_and_conflicts(tmp_path):
    proposals = tmp_path / "props.jsonl"
    silver = tmp_path / "silver.jsonl"
    windows = tmp_path / "windows.jsonl"
    pairs = tmp_path / "pairs.jsonl"
    rows = [
        _proposal("win1", "cowgirl_fast_shallow", 0.9),
        _proposal("win1", "cowgirl_fast_shallow", 0.9),
        _proposal("win1", "cowgirl_deep_slow", 0.85),
        _proposal("win1", "rider_active", 0.85, "role_candidate", "pair1"),
        _proposal("win1", "partner_context_static", 0.82, "role_candidate", "pair2"),
    ]
    write_jsonl(proposals, rows)
    write_jsonl(silver, [{"window_id": "win1", "pair_window_id": None, "positive_labels": ["cowgirl_fast_shallow"], "role_candidates": ["rider_active"], "confidence_by_label": {"cowgirl_fast_shallow": 0.9}, "is_human_ground_truth": False}])
    write_jsonl(windows, [{"window_id": "win1"}])
    write_jsonl(pairs, [{"pair_window_id": "pair1"}, {"pair_window_id": "pair2"}])

    summary = audit_machine_labels_v1(tmp_path, proposals, silver, windows, pairs, tmp_path / "audit.md", tmp_path / "audit.json")

    assert summary["duplicate_proposal_key_count"] == 1
    assert summary["conflict_counts"]["fast_and_slow"] >= 1
    assert summary["conflict_counts"]["rider_active_and_partner_context_static"] >= 1


def test_aggregation_collapses_pair_context_inflation(tmp_path):
    proposals = tmp_path / "props.jsonl"
    rows = [_proposal("win1", "cowgirl_vertical_bounce", 0.86, pair_window_id=f"pair{i}", rule="pair_rule", source="machine_pair_rule_v1") for i in range(20)]
    rows += [_proposal("win1", "cowgirl_vertical_bounce", 0.9, pair_window_id=None, rule="window_rule")]
    write_jsonl(proposals, rows)

    summary = aggregate_machine_labels_v2(proposals, tmp_path / "window_scores.jsonl", tmp_path / "pair_scores.jsonl", tmp_path / "agg.md")
    window_scores = load_jsonl(tmp_path / "window_scores.jsonl")

    assert summary["window_score_rows"] == 1
    assert window_scores[0]["supporting_pair_window_count"] == 20
    assert window_scores[0]["final_score"] <= 1.0
    assert window_scores[0]["recommended_status"] == "silver_positive_candidate"


def test_silver_v2_uses_scores_and_excludes_contact_unknown(tmp_path):
    window_scores = tmp_path / "window_scores.jsonl"
    pair_scores = tmp_path / "pair_scores.jsonl"
    write_jsonl(window_scores, [
        {"scope": "window", "window_id": "win1", "window_ids": ["win1"], "sample_ids": ["s1"], "label": "cowgirl_vertical_bounce", "final_score": 0.88, "max_confidence": 0.9, "recommended_status": "silver_positive_candidate", "rule_ids": ["r"], "evidence_count": 2, "distinct_rule_count": 1, "supporting_pair_window_count": 0, "source_scene_files": ["scene.json"], "technical_atom_ids": ["Actor"]},
        {"scope": "window", "window_id": "win2", "window_ids": ["win2"], "sample_ids": ["s2"], "label": "contact_unknown", "final_score": 0.95, "max_confidence": 0.95, "recommended_status": "review_only", "rule_ids": ["r"], "evidence_count": 2, "distinct_rule_count": 1, "supporting_pair_window_count": 0, "source_scene_files": ["scene.json"], "technical_atom_ids": ["Actor"]},
    ])
    write_jsonl(pair_scores, [])

    summary = build_silver_labels_v2(window_scores, pair_scores, tmp_path / "silver_win.jsonl", tmp_path / "silver_pair.jsonl", tmp_path / "silver.yaml", tmp_path / "silver.md", min_score=0.78)
    rows = load_jsonl(tmp_path / "silver_win.jsonl")

    assert summary["v2_silver_window_records"] == 2
    by_window = {row["window_id"]: row for row in rows}
    assert by_window["win1"]["positive_labels"] == ["cowgirl_vertical_bounce"]
    assert "contact_unknown" not in by_window["win2"]["positive_labels"]
    assert by_window["win2"]["review_only_labels"] == ["contact_unknown"]
    assert all(row["is_human_ground_truth"] is False for row in rows)


def test_dataset_v4_separates_labels_and_excludes_role_defaults(tmp_path):
    features = tmp_path / "features.jsonl"
    windows = tmp_path / "windows.jsonl"
    weak = tmp_path / "weak.jsonl"
    manual = tmp_path / "manual.yaml"
    silver_win = tmp_path / "silver_win.jsonl"
    silver_pair = tmp_path / "silver_pair.jsonl"
    write_jsonl(features, [{"window_id": "win1", "sample_id": "s1", "source_id": "src1", "source_scene_file": "scene.json", "technical_atom_id": "Actor", "feature_values": {"pelvis_movement_energy": 1.0}, "feature_quality": {"has_pelvis_features": True}}])
    write_jsonl(windows, [{"window_id": "win1", "sample_id": "s1", "source_id": "src1", "source_scene_file": "scene.json", "technical_atom_id": "Actor"}])
    write_jsonl(weak, [{"window_id": "win1", "weak_labels": [{"label": "weak_v2_high_vertical_motion"}]}])
    manual.write_text("windows: {}\n", encoding="utf-8")
    write_jsonl(silver_win, [{"window_id": "win1", "positive_labels": ["cowgirl_vertical_bounce"], "negative_labels": [], "scores_by_label": {"cowgirl_vertical_bounce": 0.9}, "excluded_from_default_training": {}, "label_source": "silver_machine_v2", "is_human_ground_truth": False}])
    write_jsonl(silver_pair, [{"pair_window_id": "pair1", "window_ids": ["win1"], "positive_labels": ["rider_active"], "negative_labels": [], "scores_by_label": {"rider_active": 0.9}, "excluded_from_default_training": {"rider_active": "high-risk role proxy label"}, "label_source": "silver_machine_v2", "is_human_ground_truth": False}])

    summary = build_ml_dataset_v4(features, windows, weak, manual, silver_win, silver_pair, tmp_path / "dataset.npz", tmp_path / "dataset.md")
    data = np.load(tmp_path / "dataset.npz", allow_pickle=True)

    assert summary["shape"] == [1, 1]
    assert data["silver_v2_window_y_multilabel"].sum() == 1
    assert data["silver_v2_pair_y_multilabel"].sum() == 1
    excluded = data["excluded_silver_label_names"].tolist()
    assert "rider_active" in excluded
    assert data["default_trainable_silver_label_mask"].sum() == 1


def test_readiness_v2_flags_default_and_excluded_labels(tmp_path):
    dataset = tmp_path / "dataset.npz"
    labels = np.asarray(["cowgirl_vertical_bounce", "rider_active"], dtype=object)
    np.savez_compressed(
        dataset,
        silver_label_names=labels,
        silver_window_label_names=np.asarray(["cowgirl_vertical_bounce"], dtype=object),
        silver_pair_label_names=np.asarray(["rider_active"], dtype=object),
        silver_v2_window_y_multilabel=np.asarray([[1], [0], [1], [0]], dtype=np.int8),
        silver_v2_pair_y_multilabel=np.asarray([[1], [1], [0], [0]], dtype=np.int8),
        group_scene=np.asarray(["a", "b", "c", "d"], dtype=object),
        group_sample=np.asarray(["s1", "s2", "s3", "s4"], dtype=object),
        default_trainable_silver_label_mask=np.asarray([True, False]),
        excluded_silver_label_names=np.asarray(["rider_active"], dtype=object),
        exclusion_reasons=np.asarray(["high-risk role proxy label"], dtype=object),
    )
    write_jsonl(tmp_path / "sw.jsonl", [{"window_id": "w1"}])
    write_jsonl(tmp_path / "sp.jsonl", [{"pair_window_id": "p1"}])

    summary = analyze_silver_readiness_v2(dataset, tmp_path / "sw.jsonl", tmp_path / "sp.jsonl", tmp_path / "ready.md", min_positive=2, min_scenes=2, min_samples=2)

    assert "cowgirl_vertical_bounce" in summary["labels_trainable_by_default"]
    assert "rider_active" in summary["labels_excluded_high_risk"]


def test_silver_baseline_v1_uses_grouped_numpy_or_sklearn_proxy(tmp_path):
    n = 120
    scenes = np.asarray([f"scene_{i % 6}" for i in range(n)], dtype=object)
    samples = np.asarray([f"sample_{i % 24}" for i in range(n)], dtype=object)
    y = np.asarray([[1 if i % 2 == 0 else 0] for i in range(n)], dtype=np.int8)
    X = np.asarray([[float(i % 2), float(i % 5)] for i in range(n)], dtype=np.float32)
    dataset = tmp_path / "dataset.npz"
    np.savez_compressed(
        dataset,
        X=X,
        feature_names=np.asarray(["signal", "noise"], dtype=object),
        silver_label_names=np.asarray(["cowgirl_vertical_bounce"], dtype=object),
        silver_window_label_names=np.asarray(["cowgirl_vertical_bounce"], dtype=object),
        silver_v2_window_y_multilabel=y,
        group_scene=scenes,
        group_sample=samples,
        default_trainable_silver_label_mask=np.asarray([True]),
    )

    summary = train_silver_baseline_v1(dataset, tmp_path / "ready.md", tmp_path / "models", tmp_path / "baseline.md", allow_numpy_fallback=True)
    text = (tmp_path / "baseline.md").read_text(encoding="utf-8")

    assert summary["trained"] is True
    assert summary["sklearn_used"] or summary["numpy_fallback_used"]
    assert "not human-supervised" in text
    assert "proxy" in text


def test_machine_review_batch_v2_stub_empty_and_role_limited(tmp_path):
    run = tmp_path / "run"
    (run / "semantic").mkdir(parents=True)
    (run / "features").mkdir()
    rows = []
    feats = []
    weak = []
    for i in range(30):
        wid = f"win{i}"
        rows.append({"window_id": wid, "sample_id": f"sample{i}", "source_scene_file": f"scene{i % 5}.json", "technical_atom_id": "Actor", "start_seconds": 0, "end_seconds": 2})
        feats.append({"window_id": wid, "sample_id": f"sample{i}", "source_scene_file": f"scene{i % 5}.json", "technical_atom_id": "Actor", "feature_values": {"pelvis_movement_energy": 1.0}})
        weak.append({"window_id": wid, "weak_labels": []})
    write_jsonl(run / "semantic" / "movement_windows.jsonl", rows)
    write_jsonl(run / "features" / "cowgirl_window_features_v1.jsonl", feats)
    write_jsonl(run / "semantic" / "weak_labels_v2.jsonl", weak)
    score_rows = []
    for i in range(20):
        score_rows.append({"window_id": f"win{i}", "window_ids": [f"win{i}"], "label": "rider_active", "final_score": 0.95, "recommended_status": "silver_positive_candidate", "rule_ids": ["r"], "conflict_flags": [], "high_risk_proxy_label": True})
    for i in range(20, 30):
        score_rows.append({"window_id": f"win{i}", "window_ids": [f"win{i}"], "label": "cowgirl_vertical_bounce", "final_score": 0.9, "recommended_status": "silver_positive_candidate", "rule_ids": ["r"], "conflict_flags": [], "high_risk_proxy_label": False})
    write_jsonl(tmp_path / "window_scores.jsonl", score_rows)
    write_jsonl(tmp_path / "pair_scores.jsonl", [])
    write_jsonl(tmp_path / "silver_win.jsonl", [{"window_id": row["window_id"], "positive_labels": [row["label"]], "scores_by_label": {row["label"]: row["final_score"]}} for row in score_rows])
    write_jsonl(tmp_path / "silver_pair.jsonl", [])

    selected = build_machine_proposal_review_batch_v2(run, tmp_path / "window_scores.jsonl", tmp_path / "pair_scores.jsonl", tmp_path / "silver_win.jsonl", tmp_path / "silver_pair.jsonl", tmp_path / "batch", batch_size=24)
    stub = yaml.safe_load((tmp_path / "batch" / "manual_labels.stub.yaml").read_text(encoding="utf-8"))
    role_count = sum(1 for row in selected for item in row.get("machine_scores_v2", []) if item.get("label") == "rider_active")

    assert selected
    assert all(entry["labels"] == [] for entry in stub["windows"].values())
    assert role_count <= 10
