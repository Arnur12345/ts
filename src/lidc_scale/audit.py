from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from .core import canonical_json, sha256_file


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def record_audit(
    stimulus_dir: Path, reviewer: str, decision: str, notes: str
) -> dict[str, Any]:
    review_path = stimulus_dir / "audit" / "review.json"
    if review_path.exists():
        raise ValueError(f"refusing to replace existing audit record: {review_path}")
    contact_sheet = stimulus_dir / "audit" / "contact_sheet_20.png"
    audit_rows = _read_jsonl(stimulus_dir / "audit" / "audit_manifest.jsonl")
    if not contact_sheet.is_file() or len(audit_rows) != 20:
        raise ValueError("audit sheet or its 20-row manifest is missing")
    if len({row["nodule_id"] for row in audit_rows}) != 20:
        raise ValueError("audit manifest must contain 20 distinct nodules")
    for row in audit_rows:
        image_path = stimulus_dir / row["marker_image"]
        if not image_path.is_file() or sha256_file(image_path) != row["marker_sha256"]:
            raise ValueError(f"missing or modified audit image: {image_path}")
    review = {
        "decision": decision,
        "reviewer": reviewer,
        "notes": notes,
        "reviewed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "contact_sheet": "audit/contact_sheet_20.png",
        "contact_sheet_sha256": sha256_file(contact_sheet),
        "checks": [
            "marker centered on intended nodule",
            "lung window visually valid",
            "orientation and anatomy valid",
            "nodule footprint responds to target spacing",
        ],
    }
    review_path.write_text(canonical_json(review) + "\n", encoding="utf-8")
    summary_path = stimulus_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["status"] = "audit_passed" if decision == "pass" else "audit_failed"
    summary["audit_record"] = "audit/review.json"
    summary_path.write_text(canonical_json(summary) + "\n", encoding="utf-8")
    return review


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Record the required human review of 20 LIDC stimuli")
    parser.add_argument("stimulus_dir", type=Path)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--decision", choices=("pass", "fail"), required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)
    review = record_audit(args.stimulus_dir, args.reviewer, args.decision, args.notes)
    print(json.dumps(review, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
