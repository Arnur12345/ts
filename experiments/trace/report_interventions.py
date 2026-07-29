"""Extract high-precision chest-tube transitions from raw MIMIC-CXR reports."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
from collections import Counter
from pathlib import Path

from .core import study_timestamp, subject_bucket


TUBE = (
    r"(?:(?:left|right|bilateral)(?:-sided)?\s+)?"
    r"(?:chest|thoracostomy|pleural)\s+"
    r"(?:tube|catheter|drain)s?"
)
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
INSERTED = (
    re.compile(
        rf"\b(?:interval|new|newly|recent)\s+"
        rf"(?:placement|insertion)?\s*(?:of\s+)?(?:a\s+)?{TUBE}\b"
    ),
    re.compile(
        rf"\b(?:placement|insertion)\s+of\s+(?:a\s+)?{TUBE}\b"
    ),
    re.compile(
        rf"\b{TUBE}\b.{{0,45}}\b(?:has\s+been\s+|was\s+)?"
        r"(?:placed|inserted)\b"
    ),
)
REMOVED = (
    re.compile(
        rf"\b(?:interval\s+)?removal\s+of\s+(?:the\s+|a\s+)?{TUBE}\b"
    ),
    re.compile(
        rf"\b{TUBE}\b.{{0,45}}\b(?:has\s+been\s+|was\s+)?"
        r"(?:removed|withdrawn|discontinued)\b"
    ),
)
STABLE_PRESENT = (
    re.compile(
        rf"\b{TUBE}\b.{{0,55}}\b"
        r"(?:is\s+)?(?:unchanged|stable|in\s+place|remains?|persists?)\b"
    ),
    re.compile(
        rf"\b(?:unchanged|stable|remaining)\s+(?:\w+\s+){{0,3}}{TUBE}\b"
    ),
)
EXPLICIT_ABSENT = (
    re.compile(rf"\bno\s+(?:indwelling\s+)?{TUBE}\b"),
    re.compile(rf"\bwithout\s+(?:an?\s+)?{TUBE}\b"),
    re.compile(rf"\b{TUBE}\b.{{0,25}}\b(?:is\s+)?absent\b"),
)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _matching_sentences(text: str, patterns) -> list[str]:
    matched = []
    for sentence in SENTENCE.split(text):
        normalised = _normalise(sentence)
        if any(pattern.search(normalised) for pattern in patterns):
            matched.append(normalised)
    return matched


def extract_report_evidence(text: str) -> dict[str, list[str]]:
    """Return explicit report evidence; missing mention remains unknown."""

    return {
        "inserted": _matching_sentences(text, INSERTED),
        "removed": _matching_sentences(text, REMOVED),
        "stable_present": _matching_sentences(text, STABLE_PRESENT),
        "explicit_absent": _matching_sentences(text, EXPLICIT_ABSENT),
    }


def classify_transition(
    before_report: str, after_report: str
) -> tuple[str, str, str] | None:
    """Classify one raw-consecutive transition, dropping ambiguous evidence."""

    before = extract_report_evidence(before_report)
    after = extract_report_evidence(after_report)
    direct = [
        name
        for name in ("inserted", "removed", "stable_present")
        if after[name]
    ]
    if len(direct) == 1:
        name = direct[0]
        return name, f"after_report_{name}", after[name][-1]
    if direct:
        return None
    if before["explicit_absent"] and after["explicit_absent"]:
        return (
            "stable_absent",
            "explicit_absence_in_both_reports",
            after["explicit_absent"][-1],
        )
    return None


def _open_csv(path: Path):
    return (
        gzip.open(path, "rt", encoding="utf-8", newline="")
        if path.suffix == ".gz"
        else path.open("r", encoding="utf-8", newline="")
    )


def _normalise_id(value) -> str:
    raw = str(value).strip()
    try:
        return str(int(float(raw)))
    except ValueError:
        return raw


def _canonical_rows(manifest: Path, metadata: Path) -> list[dict]:
    with _open_csv(manifest) as handle:
        rows = list(csv.DictReader(handle))
    with _open_csv(metadata) as handle:
        metadata_rows = list(csv.DictReader(handle))
    if not rows or not metadata_rows:
        raise ValueError("canonical manifest or metadata CSV is empty")
    by_dicom = {
        str(row.get("dicom_id", "")).strip(): row
        for row in metadata_rows
    }
    matched = 0
    for row in rows:
        row["subject_id"] = _normalise_id(row.get("subject_id", ""))
        row["study_id"] = _normalise_id(row.get("study_id", ""))
        extra = by_dicom.get(str(row.get("dicom_id", "")).strip())
        if extra is None:
            continue
        matched += 1
        for field in ("StudyDate", "StudyTime", "ViewPosition"):
            if not str(row.get(field, "")).strip():
                row[field] = extra.get(field, "")
    if not matched:
        raise ValueError("metadata did not match the canonical manifest")
    return rows


def _report_path(
    root: Path, subject_id: str, study_id: str
) -> Path | None:
    padded = f"{int(subject_id):08d}"
    relative = Path(
        f"p{padded[:2]}/p{padded}/s{int(study_id)}.txt"
    )
    candidates = (
        root / relative,
        root / "files" / relative,
        root / f"s{int(study_id)}.txt",
        root / relative.with_suffix(".txt.gz"),
        root / "files" / relative.with_suffix(".txt.gz"),
    )
    return next((path for path in candidates if path.is_file()), None)


def _read_report(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8", errors="replace")


def _write(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build(args: argparse.Namespace) -> None:
    rows = _canonical_rows(args.canonical_manifest, args.metadata_csv)
    rows = [
        row
        for row in rows
        if subject_bucket(row["subject_id"], args.split_seed, "iera") < 7000
    ]
    if not rows:
        raise ValueError("canonical manifest contains no main-training patients")
    grouped: dict[str, list[tuple[tuple[int, float], dict]]] = {}
    for row in rows:
        timestamp = study_timestamp(row)
        if timestamp is None:
            continue
        grouped.setdefault(row["subject_id"], []).append((timestamp, row))
    audit = []
    missing_reports = 0
    raw_pairs = 0
    for subject, studies in grouped.items():
        studies.sort(
            key=lambda item: (
                item[0],
                item[1]["study_id"],
                item[1].get("dicom_id", ""),
            )
        )
        for (_, before), (_, after) in zip(studies, studies[1:]):
            raw_pairs += 1
            before_path = _report_path(
                args.reports_root, subject, before["study_id"]
            )
            after_path = _report_path(
                args.reports_root, subject, after["study_id"]
            )
            if before_path is None or after_path is None:
                missing_reports += 1
                continue
            classified = classify_transition(
                _read_report(before_path), _read_report(after_path)
            )
            if classified is None:
                continue
            transition, rule, evidence = classified
            audit.append(
                {
                    "subject_id": subject,
                    "before_study_id": before["study_id"],
                    "after_study_id": after["study_id"],
                    "chest_tube_transition": transition,
                    "rule": rule,
                    "evidence": evidence,
                    "before_report": str(before_path),
                    "after_report": str(after_path),
                }
            )
    if raw_pairs == 0:
        raise ValueError("canonical manifest contains no chronological pairs")
    if missing_reports == raw_pairs:
        raise FileNotFoundError(
            "no MIMIC reports were found under --reports-root; point it at "
            "the MIMIC-CXR report tree containing files/pXX/pXXXXXXXX/"
            "sXXXXXXXX.txt"
        )
    if not audit:
        raise ValueError(
            "reports were found but conservative rules extracted no "
            "chest-tube transitions"
        )
    transitions = [
        {
            "before_study_id": row["before_study_id"],
            "after_study_id": row["after_study_id"],
            "chest_tube_transition": row["chest_tube_transition"],
        }
        for row in audit
    ]
    _write(
        args.output,
        transitions,
        [
            "before_study_id",
            "after_study_id",
            "chest_tube_transition",
        ],
    )
    _write(args.audit_output, audit, list(audit[0]))
    review = []
    rng = random.Random(args.seed)
    for transition in (
        "inserted",
        "removed",
        "stable_present",
        "stable_absent",
    ):
        candidates = [
            dict(row)
            for row in audit
            if row["chest_tube_transition"] == transition
        ]
        rng.shuffle(candidates)
        for row in candidates[: args.review_per_transition]:
            row["reviewer_label"] = ""
            row["approved"] = ""
            review.append(row)
    _write(
        args.review_output,
        review,
        list(review[0]) if review else [*audit[0], "reviewer_label", "approved"],
    )
    summary = {
        "raw_consecutive_pairs": raw_pairs,
        "pairs_missing_one_or_both_reports": missing_reports,
        "extracted_transitions": len(transitions),
        "transition_counts": dict(
            sorted(Counter(row["chest_tube_transition"] for row in audit).items())
        ),
        "method": "conservative_rule_based_MIMIC_report_extraction",
        "patient_scope": "deterministic main-training partition only",
        "split_seed": args.split_seed,
        "unknown_and_ambiguous_policy": "drop",
        "manual_review_required_for_paper": True,
        "review_rows": len(review),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {len(transitions):,} chest-tube transitions to {args.output}",
        flush=True,
    )
    print(
        f"manual audit sample written to {args.review_output}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/trace/chest_tube_transitions.csv"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("outputs/trace/chest_tube_transition_audit.csv"),
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        default=Path("outputs/trace/chest_tube_transition_review.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/trace/chest_tube_transition_summary.json"),
    )
    parser.add_argument("--review-per-transition", type=int, default=50)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.review_per_transition <= 0:
        parser.error("review-per-transition must be positive")
    build(args)


if __name__ == "__main__":
    main()
