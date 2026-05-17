"""Evaluate the ML-assisted review batch after human answers arrive."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from vam_timeline_ai.audits.review_answer_ingestion import _derive_from_answer, _derive_verdict
from vam_timeline_ai.io.json_utils import load_jsonl


def evaluate_ml_assisted_review_v1(
    review_dir: str | Path,
    model_scores: str | Path,
    answers: str | Path,
    out: str | Path,
) -> dict[str, Any]:
    review = Path(review_dir)
    review_rows = {str(r.get("review_id")): r for r in load_jsonl(review / "semantic_review_010.jsonl")}
    answer_rows = _load_answers(Path(answers))
    score_by_window = {str(r.get("window_id")): r for r in load_jsonl(model_scores)}
    bucket_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "human_correct": 0, "human_wrong": 0, "human_unclear": 0, "tags": Counter(), "model_correct": 0, "heuristic_correct": 0})
    rows = []
    for answer in answer_rows:
        rid = str(answer.get("review_id") or "")
        system = review_rows.get(rid, {})
        bucket = str(system.get("recommended_review_priority") or "unknown")
        derived = _derive_from_answer(answer)
        verdict = answer.get("verdict") or _derive_verdict(answer)
        human_cowgirl = _human_cowgirl(answer, derived)
        model_cowgirl = _model_cowgirl(system, score_by_window)
        heuristic_cowgirl = str(system.get("category") or "").startswith("cowgirl_") or str(system.get("semantic_family") or "") == "cowgirl"
        stat = bucket_stats[bucket]
        stat["count"] += 1
        if verdict == "correct":
            stat["human_correct"] += 1
        elif verdict == "wrong":
            stat["human_wrong"] += 1
        else:
            stat["human_unclear"] += 1
        if human_cowgirl is not None and model_cowgirl is not None and human_cowgirl == model_cowgirl:
            stat["model_correct"] += 1
        if human_cowgirl is not None and human_cowgirl == heuristic_cowgirl:
            stat["heuristic_correct"] += 1
        for tag in (answer.get("error_tags") or []) + (derived.get("error_tags") or []):
            stat["tags"][str(tag)] += 1
        rows.append({"review_id": rid, "bucket": bucket, "verdict": verdict, "human_cowgirl": human_cowgirl, "model_cowgirl": model_cowgirl, "heuristic_cowgirl": heuristic_cowgirl})
    summary = {
        "status": "ok",
        "answers": len(answer_rows),
        "bucket_stats": {
            bucket: {
                "item_count": stat["count"],
                "human_correct_count": stat["human_correct"],
                "human_wrong_count": stat["human_wrong"],
                "human_unclear_count": stat["human_unclear"],
                "precision_proxy": stat["human_correct"] / stat["count"] if stat["count"] else None,
                "model_correct_count": stat["model_correct"],
                "heuristic_correct_count": stat["heuristic_correct"],
                "common_error_tags": dict(stat["tags"].most_common(10)),
            }
            for bucket, stat in bucket_stats.items()
        },
        "ml_v1_helped": _helped(bucket_stats),
        "rows": rows,
    }
    _write_report(out, summary)
    return summary


def _load_answers(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return [dict({"review_id": rid}, **(row or {})) for rid, row in (data.get("reviews") or {}).items()]
    return load_jsonl(path)


def _human_cowgirl(answer: dict[str, Any], derived: dict[str, Any]) -> bool | None:
    family = str(answer.get("actual_semantic_family") or derived.get("semantic_family") or "").lower()
    labels = {str(x).lower() for x in answer.get("review_labels") or []}
    notes = str(answer.get("notes") or "").lower()
    if family == "cowgirl" or labels & {"correct_clean_cowgirl_motion", "correct_short_cowgirl_motion", "correct_lean_back_supported_cowgirl"}:
        return True
    if family in {"bj_oral", "standing_hand_head_gesture", "receiver_response", "unknown"} or labels & {"bj_oral_not_cowgirl", "standing_hand_head_not_cowgirl", "receiver_response_not_rider_motion", "broken_pose_or_bad_data"}:
        return False
    if "not cowgirl" in notes or "nicht cowgirl" in notes:
        return False
    if "cowgirl" in notes:
        return True
    return None


def _model_cowgirl(system: dict[str, Any], score_by_window: dict[str, dict[str, Any]]) -> bool | None:
    prob = system.get("model_cowgirl_probability")
    if prob is None:
        prob = (score_by_window.get(str(system.get("window_id") or "")) or {}).get("model_cowgirl_probability")
    if prob is None:
        return None
    return float(prob) >= 0.5


def _helped(bucket_stats: dict[str, dict[str, Any]]) -> str:
    total = sum(stat["count"] for stat in bucket_stats.values())
    if not total:
        return "unknown"
    model = sum(stat["model_correct"] for stat in bucket_stats.values())
    heuristic = sum(stat["heuristic_correct"] for stat in bucket_stats.values())
    if model > heuristic:
        return "yes"
    if model < heuristic:
        return "no"
    return "uncertain"


def _write_report(path: str | Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ML-assisted Cowgirl Review v1 Evaluation",
        "",
        f"- Answers evaluated: {summary['answers']}",
        f"- ML v1 helped: `{summary['ml_v1_helped']}`",
        "",
        "## Buckets",
        "",
    ]
    for bucket, stat in summary["bucket_stats"].items():
        lines.extend(
            [
                f"### {bucket}",
                f"- Items: {stat['item_count']}",
                f"- Human correct/wrong/unclear: {stat['human_correct_count']} / {stat['human_wrong_count']} / {stat['human_unclear_count']}",
                f"- Precision proxy: `{stat['precision_proxy']}`",
                f"- Model correct count: {stat['model_correct_count']}",
                f"- Heuristic correct count: {stat['heuristic_correct_count']}",
                f"- Common error tags: `{stat['common_error_tags']}`",
                "",
            ]
        )
    lines.extend(["## Recommendation", "", "- Use this evaluation to adjust v2 sampling buckets; do not auto-label unreviewed items."])
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
