"""Sourcebook ingestion for the Semantik master ontology document."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json
import zipfile
import xml.etree.ElementTree as ET


def ingest_semantik_sourcebook_v2(source_docx: str | Path, out_dir: str | Path, report: str | Path) -> dict[str, Any]:
    """Extract the DOCX sourcebook text and write a trace report.

    This records the canonical semantic source. It does not train, auto-label,
    generate Timeline animation, or modify manual label files.
    """

    src = Path(source_docx)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        summary = {"status": "blocked", "reason": "source_docx_missing", "source_docx": str(src)}
        _write_report(Path(report), summary)
        return summary

    paragraphs = _extract_docx_paragraphs(src)
    text_path = out / "semantik_master_konsolidiert_extracted_v1.txt"
    manifest_path = out / "semantik_master_konsolidiert_manifest_v1.json"
    text_path.write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")
    manifest = {
        "source_path": str(src),
        "source_size_bytes": src.stat().st_size,
        "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "paragraph_count": len(paragraphs),
        "extracted_text_path": str(text_path),
        "extraction_method": "python_zipfile_word_document_xml",
        "canonical_use": "motion_ontology_sourcebook",
        "root_mapping": "root/root-node in sourcebook maps to pelvisControl/hipControl/abdomen region, never VaM Person/root/world",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "status": "ok",
        "paragraph_count": len(paragraphs),
        "manifest": str(manifest_path),
        "text": str(text_path),
        "sha256": manifest["sha256"],
    }
    _write_report(Path(report), summary)
    return summary


def _extract_docx_paragraphs(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        document_xml = zf.read("word/document.xml")
    root = ET.fromstring(document_xml)
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs: list[str] = []
    for para in root.iter(ns + "p"):
        pieces = [node.text or "" for node in para.iter(ns + "t")]
        text = "".join(pieces).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Semantik Master Sourcebook Ingestion V2",
        "",
        "Canonical source for the motion ontology is the user-provided DOCX.",
        "",
        f"- Status: {summary.get('status')}",
        f"- Paragraphs: {summary.get('paragraph_count', 0)}",
        f"- Manifest: `{summary.get('manifest', '')}`",
        f"- Extracted text: `{summary.get('text', '')}`",
        f"- SHA256: `{summary.get('sha256', '')}`",
        "- No ML training: true",
        "- No auto-labeling: true",
        "- manual_labels.yaml modified: false",
        "- Root mapping: sourcebook root/root-node means pelvis/hip/abdomen controls, never VaM Person/root/world.",
    ]
    if summary.get("status") == "blocked":
        lines.append(f"- Blocked reason: {summary.get('reason')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
