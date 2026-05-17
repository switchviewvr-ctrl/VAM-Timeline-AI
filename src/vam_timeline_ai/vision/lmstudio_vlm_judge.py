"""Local LM Studio VLM adapter for visual judge requests."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import base64
import json
import time
import urllib.parse
import urllib.request

from vam_timeline_ai.io.json_utils import dump_json, load_jsonl, write_jsonl
from vam_timeline_ai.vision.visual_judge_schema import parse_json_from_text_response, validate_visual_judge_result


def run_lmstudio_vlm_judge_v0(
    requests: str | Path,
    base_url: str,
    model: str,
    out_jsonl: str | Path,
    out_raw_dir: str | Path,
    dry_run: bool = True,
) -> dict[str, Any]:
    rows = load_jsonl(requests)
    raw = Path(out_raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    out_rows = []
    if dry_run:
        for row in rows:
            payload = _payload_for(row, model, include_image=False)
            raw_path = raw / f"{row.get('review_id')}_planned_payload.json"
            dump_json(raw_path, payload)
            out_rows.append(
                validate_visual_judge_result(
                    {
                        "item_id": row.get("review_id"),
                        "review_id": row.get("review_id"),
                        "visual_model_name": model,
                        "backend": "lmstudio",
                        "visual_input_path": row.get("primary_visual_path"),
                        "visual_input_type": row.get("primary_visual_type"),
                        "visual_quality": row.get("visual_quality"),
                        "parse_status": "dry_run",
                        "warnings": ["dry-run only; LM Studio was not called"],
                    },
                    enforce_rules=True,
                )
            )
        write_jsonl(out_jsonl, out_rows)
        return {"status": "dry_run", "requests": len(rows), "out_jsonl": str(out_jsonl), "raw_dir": str(raw), "model": model}
    if not _is_local_base_url(base_url):
        blocked = Path(out_jsonl).parent / "BLOCKED_LMSTUDIO_UNAVAILABLE.md"
        blocked.write_text("# LM Studio blocked\n\nOnly localhost/127.0.0.1 base URLs are allowed.\n", encoding="utf-8")
        write_jsonl(out_jsonl, [])
        return {"status": "blocked", "reason": "non-local base_url", "blocked_report": str(blocked)}
    try:
        _check_models(base_url)
    except Exception as exc:  # noqa: BLE001
        blocked = Path(out_jsonl).parent / "BLOCKED_LMSTUDIO_UNAVAILABLE.md"
        blocked.write_text(f"# LM Studio unavailable\n\n- Base URL: `{base_url}`\n- Error: `{exc}`\n", encoding="utf-8")
        write_jsonl(out_jsonl, [])
        return {"status": "blocked", "reason": str(exc), "blocked_report": str(blocked)}
    for row in rows:
        raw_path = raw / f"{row.get('review_id')}_raw.json"
        try:
            payload = _payload_for(row, model, include_image=True)
            dump_json(raw / f"{row.get('review_id')}_payload.json", payload)
            started = time.perf_counter()
            response_body = _chat(base_url, payload)
            latency_seconds = round(time.perf_counter() - started, 3)
            dump_json(raw_path, response_body)
            response_text = _message_text(response_body)
            parsed, error = parse_json_from_text_response(response_text)
            if parsed is None:
                failed = _failed(row, model, raw_path, error)
                failed["latency_seconds"] = latency_seconds
                failed["actual_model_name"] = response_body.get("model")
                out_rows.append(failed)
            else:
                parsed.update({"item_id": row.get("review_id"), "review_id": row.get("review_id"), "visual_model_name": model, "backend": "lmstudio", "visual_input_path": row.get("primary_visual_path"), "visual_input_type": row.get("primary_visual_type"), "visual_quality": row.get("visual_quality"), "raw_response_path": str(raw_path), "parse_status": "parsed", "latency_seconds": latency_seconds, "actual_model_name": response_body.get("model")})
                out_rows.append(validate_visual_judge_result(parsed, enforce_rules=True))
        except Exception as exc:  # noqa: BLE001
            out_rows.append(_failed(row, model, raw_path, str(exc)))
        write_jsonl(out_jsonl, out_rows)
    write_jsonl(out_jsonl, out_rows)
    latencies = [float(r["latency_seconds"]) for r in out_rows if r.get("latency_seconds") is not None]
    return {"status": "ok", "requests": len(rows), "parsed": sum(1 for r in out_rows if r.get("parse_status") == "parsed"), "parse_failed": sum(1 for r in out_rows if r.get("parse_status") == "parse_failed"), "average_latency_seconds": round(sum(latencies) / len(latencies), 3) if latencies else None, "out_jsonl": str(out_jsonl), "raw_dir": str(raw)}


def _payload_for(row: dict[str, Any], model: str, include_image: bool) -> dict[str, Any]:
    prompt = row.get("prompt_text") or ""
    if "/no_think" not in prompt[:64].lower():
        prompt = "/no_think\n" + prompt
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    path = row.get("primary_visual_path")
    if include_image and path and Path(str(path)).exists():
        mime = "image/png"
        if str(path).lower().endswith(".gif"):
            mime = "image/gif"
        elif str(path).lower().endswith(".mp4"):
            mime = "video/mp4"
        b64 = base64.b64encode(Path(str(path)).read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    else:
        content.append({"type": "text", "text": f"Visual path: {path}"})
    return {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0.0, "max_tokens": 4096}


def _is_local_base_url(base_url: str) -> bool:
    parsed = urllib.parse.urlparse(base_url)
    return parsed.hostname in {"localhost", "127.0.0.1", "::1"}


def _check_models(base_url: str) -> None:
    with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=3.0) as response:
        response.read()


def _chat(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode("utf-8"), headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180.0) as response:
        return json.loads(response.read().decode("utf-8"))


def _message_text(body: dict[str, Any]) -> str:
    message = body.get("choices", [{}])[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                parts.append(str(part))
        return "\n".join(p for p in parts if p)
    return str(content or "")


def _failed(row: dict[str, Any], model: str, raw_path: Path, warning: str) -> dict[str, Any]:
    return validate_visual_judge_result(
        {
            "item_id": row.get("review_id"),
            "review_id": row.get("review_id"),
            "visual_model_name": model,
            "backend": "lmstudio",
            "visual_input_path": row.get("primary_visual_path"),
            "visual_input_type": row.get("primary_visual_type"),
            "visual_quality": row.get("visual_quality"),
            "raw_response_path": str(raw_path),
            "parse_status": "parse_failed",
            "warnings": [warning],
        }
    )
