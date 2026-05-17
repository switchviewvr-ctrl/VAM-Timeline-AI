import csv
import json

from vam_timeline_ai.io.json_utils import load_json
from vam_timeline_ai.runs.sanitize_scene_identifiers import sanitize_run_scene_identifiers


def test_sanitize_scene_identifiers_replaces_scene_fields_and_keeps_aliases_idempotent(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "records.jsonl").write_text(
        json.dumps({"source_scene_path": "private/batch/real_scene.json", "safe": "keep"}) + "\n",
        encoding="utf-8",
    )
    (run / "meta.json").write_text(
        json.dumps({"scene_name": "real_scene.json", "nested": {"source_scene_file": "real_scene.json"}}),
        encoding="utf-8",
    )
    with (run / "table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["window_id", "source_scene_path"])
        writer.writeheader()
        writer.writerow({"window_id": "w1", "source_scene_path": "private/batch/other_scene.json"})

    alias_map = run / "local_scene_aliases.private.json"
    report = run / "report.md"
    summary = sanitize_run_scene_identifiers(run, alias_map, report)

    assert summary["changed_files"] == 3
    assert "real_scene" not in (run / "records.jsonl").read_text(encoding="utf-8")
    assert "other_scene" not in (run / "table.csv").read_text(encoding="utf-8")
    assert load_json(alias_map)["aliases"]["real_scene.json"].startswith("scene_")

    second = sanitize_run_scene_identifiers(run, alias_map, report)
    assert second["changed_files"] == 0
