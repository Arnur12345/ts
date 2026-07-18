"""Build and validate fixed, patient-disjoint MIMIC-CXR episodes.

The builder uses only official MIMIC-CXR metadata, CheXpert labels, and split
files.  It never derives labels from report text.  Episodes contain stable
sample identifiers and relative image paths so every evaluated method can
consume exactly the same support/query observations.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


REQUIRED_PARTITIONS = ("base", "validation_novel", "test_novel")
OFFICIAL_SPLITS = {"train", "validate", "test"}


@dataclass(frozen=True, slots=True)
class Sample:
    """One canonical frontal image representing one labeled study."""

    dicom_id: str
    study_id: str
    subject_id: str
    official_split: str
    view: str
    relative_path: str
    class_name: str

    def episode_item(self) -> dict[str, str]:
        return {
            "dicom_id": self.dicom_id,
            "study_id": self.study_id,
            "subject_id": self.subject_id,
            "relative_path": self.relative_path,
            "view": self.view,
        }


class StableRNG:
    """Small SHA-256 counter RNG with version-independent sampling behavior."""

    def __init__(self, key: str) -> None:
        self._key = key.encode("utf-8")
        self._counter = 0

    def _block(self) -> bytes:
        counter = self._counter.to_bytes(16, "big")
        self._counter += 1
        return hashlib.sha256(self._key + b"\0" + counter).digest()

    def randbelow(self, n: int) -> int:
        if n <= 0:
            raise ValueError("n must be positive")
        limit = (1 << 256) - ((1 << 256) % n)
        while True:
            value = int.from_bytes(self._block(), "big")
            if value < limit:
                return value % n

    def choice(self, values: Sequence[Any]) -> Any:
        if not values:
            raise ValueError("cannot choose from an empty sequence")
        return values[self.randbelow(len(values))]

    def sample(self, values: Sequence[Any], k: int) -> list[Any]:
        if k < 0 or k > len(values):
            raise ValueError(f"cannot sample {k} items from {len(values)}")
        result = list(values)
        for index in range(k):
            swap = index + self.randbelow(len(result) - index)
            result[index], result[swap] = result[swap], result[index]
        return result[:k]

    def shuffle(self, values: list[Any]) -> None:
        for index in range(len(values) - 1, 0, -1):
            swap = self.randbelow(index + 1)
            values[index], values[swap] = values[swap], values[index]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    _validate_config(config)
    return config


def _validate_config(config: Mapping[str, Any]) -> None:
    labels = config.get("labels")
    if not isinstance(labels, list) or len(labels) != 14 or len(set(labels)) != 14:
        raise ValueError("the protocol must define exactly 14 unique labels")

    partitions = config.get("class_partitions", {})
    expected_sizes = {"base": 8, "validation_novel": 3, "test_novel": 3}
    flattened: list[str] = []
    for name, size in expected_sizes.items():
        values = partitions.get(name)
        if not isinstance(values, list) or len(values) != size:
            raise ValueError(f"class_partitions.{name} must contain {size} classes")
        if len(set(values)) != len(values):
            raise ValueError(f"class_partitions.{name} contains duplicate classes")
        flattened.extend(values)
    if len(set(flattened)) != 14 or set(flattened) != set(labels):
        raise ValueError("base/validation_novel/test_novel must disjointly cover all labels")

    official = config.get("official_split_for_partition", {})
    if set(official) != set(REQUIRED_PARTITIONS):
        raise ValueError("official_split_for_partition must map all three partitions")
    if set(official.values()) != OFFICIAL_SPLITS:
        raise ValueError("the three partitions must use train/validate/test exactly once")

    policy = config.get("sample_policy", {})
    if policy.get("class_policy") != "exactly_one_certain_positive":
        raise ValueError("only exactly_one_certain_positive is supported in protocol v1")
    if policy.get("uncertain_label_policy") != "drop_study":
        raise ValueError("protocol v1 requires uncertain_label_policy=drop_study")
    views = policy.get("views")
    preference = policy.get("view_preference")
    if not views or set(views) != set(preference):
        raise ValueError("views and view_preference must list the same views")

    episodes = config.get("episodes", {})
    if episodes.get("ways") != 3:
        raise ValueError("protocol requires 3-way episodes")
    shots = episodes.get("shots")
    if shots != [1, 3, 5]:
        raise ValueError("protocol requires shots=[1, 3, 5]")
    if int(episodes.get("queries_per_class", 0)) <= 0:
        raise ValueError("queries_per_class must be positive")
    if int(episodes.get("episodes_per_seed", 0)) <= 0:
        raise ValueError("episodes_per_seed must be positive")
    seeds = episodes.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("protocol requires exactly five distinct seeds")
    if set(episodes.get("partitions", [])) != set(REQUIRED_PARTITIONS):
        raise ValueError("episodes.partitions must contain all three partitions")
    for required_flag in (
        "patient_disjoint_within_episode",
        "nested_shots",
        "shared_queries_across_shots",
    ):
        if episodes.get(required_flag) is not True:
            raise ValueError(f"episodes.{required_flag} must be true")

    sources = config.get("source_files", {})
    if set(sources) != {"metadata", "labels", "official_split"}:
        raise ValueError("source_files must define metadata, labels, and official_split")


def _open_text(path: Path) -> io.TextIOBase:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


@contextmanager
def _open_deterministic_gzip_text(path: Path) -> Iterator[io.TextIOWrapper]:
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
    try:
        yield text
    finally:
        text.close()
        raw.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical_json_bytes(value))


def _normalise_id(raw: Any, field: str) -> str:
    value = str(raw).strip()
    if not value:
        raise ValueError(f"missing {field}")
    try:
        return str(int(float(value)))
    except ValueError as error:
        raise ValueError(f"invalid {field}: {value!r}") from error


def _normalise_split(raw: Any) -> str:
    value = str(raw).strip().lower()
    aliases = {"validation": "validate", "val": "validate"}
    value = aliases.get(value, value)
    if value not in OFFICIAL_SPLITS:
        raise ValueError(f"unexpected official split {raw!r}")
    return value


def _parse_label(raw: Any, missing_value: int) -> int:
    value = str(raw).strip().lower()
    if value in {"", "nan", "na", "none"}:
        return missing_value
    number = float(value)
    if not math.isfinite(number) or number not in {-1.0, 0.0, 1.0}:
        raise ValueError(f"unexpected CheXpert label value {raw!r}")
    return int(number)


def _require_columns(reader: csv.DictReader, required: Iterable[str], path: Path) -> None:
    available = set(reader.fieldnames or [])
    missing = set(required) - available
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def _load_labels(
    path: Path, labels: Sequence[str], missing_value: int
) -> dict[str, tuple[str, tuple[int, ...]]]:
    by_study: dict[str, tuple[str, tuple[int, ...]]] = {}
    with _open_text(path) as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader, ["subject_id", "study_id", *labels], path)
        for row_number, row in enumerate(reader, start=2):
            subject_id = _normalise_id(row["subject_id"], "subject_id")
            study_id = _normalise_id(row["study_id"], "study_id")
            values = tuple(_parse_label(row[label], missing_value) for label in labels)
            previous = by_study.get(study_id)
            current = (subject_id, values)
            if previous is not None and previous != current:
                raise ValueError(f"conflicting label rows for study {study_id} at row {row_number}")
            by_study[study_id] = current
    return by_study


def _load_official_splits(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    by_dicom: dict[str, str] = {}
    subject_splits: dict[str, str] = {}
    with _open_text(path) as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader, ["dicom_id", "subject_id", "split"], path)
        for row in reader:
            dicom_id = str(row["dicom_id"]).strip()
            subject_id = _normalise_id(row["subject_id"], "subject_id")
            split = _normalise_split(row["split"])
            if dicom_id in by_dicom and by_dicom[dicom_id] != split:
                raise ValueError(f"DICOM {dicom_id} occurs in multiple official splits")
            by_dicom[dicom_id] = split
            previous = subject_splits.setdefault(subject_id, split)
            if previous != split:
                raise ValueError(
                    f"official split is not patient-disjoint: subject {subject_id} "
                    f"occurs in {previous} and {split}"
                )
    return by_dicom, subject_splits


def _image_relative_path(subject_id: str, study_id: str, dicom_id: str) -> str:
    padded_subject = f"{int(subject_id):08d}"
    return (
        f"files/p{padded_subject[:2]}/p{padded_subject}/"
        f"s{int(study_id)}/{dicom_id}.jpg"
    )


def _load_canonical_images(
    metadata_path: Path,
    split_by_dicom: Mapping[str, str],
    views: Sequence[str],
    view_preference: Sequence[str],
) -> dict[str, dict[str, str]]:
    priority = {view.upper(): index for index, view in enumerate(view_preference)}
    allowed = {view.upper() for view in views}
    canonical: dict[str, dict[str, str]] = {}
    study_identity: dict[str, tuple[str, str]] = {}

    with _open_text(metadata_path) as handle:
        reader = csv.DictReader(handle)
        _require_columns(
            reader, ["dicom_id", "subject_id", "study_id", "ViewPosition"], metadata_path
        )
        for row in reader:
            view = str(row["ViewPosition"]).strip().upper()
            if view not in allowed:
                continue
            dicom_id = str(row["dicom_id"]).strip()
            if not dicom_id:
                raise ValueError("metadata contains an empty dicom_id")
            subject_id = _normalise_id(row["subject_id"], "subject_id")
            study_id = _normalise_id(row["study_id"], "study_id")
            if dicom_id not in split_by_dicom:
                raise ValueError(f"DICOM {dicom_id} is absent from the official split CSV")
            official_split = split_by_dicom[dicom_id]
            identity = (subject_id, official_split)
            previous_identity = study_identity.setdefault(study_id, identity)
            if previous_identity != identity:
                raise ValueError(f"study {study_id} has inconsistent subject or split metadata")

            candidate = {
                "dicom_id": dicom_id,
                "study_id": study_id,
                "subject_id": subject_id,
                "official_split": official_split,
                "view": view,
                "relative_path": _image_relative_path(subject_id, study_id, dicom_id),
            }
            current = canonical.get(study_id)
            candidate_key = (priority[view], dicom_id)
            if current is None or candidate_key < (priority[current["view"]], current["dicom_id"]):
                canonical[study_id] = candidate
    return canonical


def _row_sort_key(row: Mapping[str, str]) -> tuple[int, int, str]:
    return (int(row["subject_id"]), int(row["study_id"]), row["dicom_id"])


def _construct_manifest(
    *,
    data_root: Path,
    labels: Sequence[str],
    label_rows: Mapping[str, tuple[str, tuple[int, ...]]],
    canonical_images: Mapping[str, Mapping[str, str]],
    check_images: bool,
) -> tuple[list[dict[str, str]], Counter[str]]:
    rows: list[dict[str, str]] = []
    exclusions: Counter[str] = Counter()
    for study_id, image in canonical_images.items():
        row = dict(image)
        class_name = ""
        eligible = "0"
        reason = ""
        label_entry = label_rows.get(study_id)
        label_values = tuple(0 for _ in labels)

        if label_entry is None:
            reason = "missing_label_row"
        else:
            label_subject, label_values = label_entry
            if label_subject != image["subject_id"]:
                raise ValueError(f"subject mismatch between metadata and labels for study {study_id}")
            positives = [label for label, value in zip(labels, label_values) if value == 1]
            if -1 in label_values:
                reason = "uncertain_label"
            elif len(positives) == 0:
                reason = "no_positive_label"
            elif len(positives) > 1:
                reason = "multiple_positive_labels"
            elif check_images and not (data_root / image["relative_path"]).is_file():
                reason = "missing_image"
            else:
                class_name = positives[0]
                eligible = "1"

        if reason:
            exclusions[reason] += 1
        row.update({"eligible": eligible, "class_name": class_name, "exclusion_reason": reason})
        row.update({label: str(value) for label, value in zip(labels, label_values)})
        rows.append(row)

    rows.sort(key=_row_sort_key)
    return rows, exclusions


def _write_manifests(
    output_dir: Path,
    rows: Sequence[Mapping[str, str]],
    labels: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[Path, Path]:
    fields = [
        "dicom_id",
        "study_id",
        "subject_id",
        "official_split",
        "view",
        "relative_path",
        "eligible",
        "class_name",
        "exclusion_reason",
        *labels,
    ]
    all_path = output_dir / "study_manifest.csv.gz"
    protocol_path = output_dir / "protocol_samples.csv.gz"
    protocol_fields = fields[:8] + ["protocol_partition"]
    class_to_partition = {
        class_name: partition
        for partition, class_names in config["class_partitions"].items()
        for class_name in class_names
    }
    official = config["official_split_for_partition"]
    with _open_deterministic_gzip_text(all_path) as all_handle, _open_deterministic_gzip_text(
        protocol_path
    ) as protocol_handle:
        all_writer = csv.DictWriter(all_handle, fieldnames=fields, lineterminator="\n")
        protocol_writer = csv.DictWriter(
            protocol_handle, fieldnames=protocol_fields, lineterminator="\n"
        )
        all_writer.writeheader()
        protocol_writer.writeheader()
        for row in rows:
            all_writer.writerow(row)
            if row["eligible"] == "1":
                partition = class_to_partition[row["class_name"]]
                if row["official_split"] == official[partition]:
                    compact_row = {field: row[field] for field in fields[:8]}
                    compact_row["protocol_partition"] = partition
                    protocol_writer.writerow(compact_row)
    return all_path, protocol_path


def _samples_by_partition_and_class(
    rows: Sequence[Mapping[str, str]], config: Mapping[str, Any]
) -> dict[str, dict[str, list[Sample]]]:
    official = config["official_split_for_partition"]
    class_to_partition = {
        class_name: partition
        for partition, class_names in config["class_partitions"].items()
        for class_name in class_names
    }
    pools: dict[str, dict[str, list[Sample]]] = {
        partition: {class_name: [] for class_name in config["class_partitions"][partition]}
        for partition in REQUIRED_PARTITIONS
    }
    for row in rows:
        if row["eligible"] != "1":
            continue
        class_name = row["class_name"]
        partition = class_to_partition[class_name]
        if row["official_split"] != official[partition]:
            continue
        pools[partition][class_name].append(
            Sample(
                dicom_id=row["dicom_id"],
                study_id=row["study_id"],
                subject_id=row["subject_id"],
                official_split=row["official_split"],
                view=row["view"],
                relative_path=row["relative_path"],
                class_name=class_name,
            )
        )
    return pools


def _group_by_subject(samples: Sequence[Sample]) -> dict[str, list[Sample]]:
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.subject_id].append(sample)
    for subject_samples in grouped.values():
        subject_samples.sort(key=lambda item: (int(item.study_id), item.dicom_id))
    return dict(grouped)


def _select_episode_samples(
    *,
    rng: StableRNG,
    class_names: Sequence[str],
    pools: Mapping[str, Sequence[Sample]],
    count_per_class: int,
    max_attempts: int = 1000,
) -> dict[str, list[Sample]]:
    by_class_subject = {name: _group_by_subject(pools[name]) for name in class_names}
    for name, grouped in by_class_subject.items():
        if len(grouped) < count_per_class:
            raise ValueError(
                f"class {name!r} has {len(grouped)} eligible patients; "
                f"at least {count_per_class} are required"
            )

    scarcity_order = sorted(class_names, key=lambda name: (len(by_class_subject[name]), name))
    for _ in range(max_attempts):
        used_subjects: set[str] = set()
        selected: dict[str, list[Sample]] = {}
        failed = False
        for class_name in scarcity_order:
            available = sorted(set(by_class_subject[class_name]) - used_subjects, key=int)
            if len(available) < count_per_class:
                failed = True
                break
            subject_ids = rng.sample(available, count_per_class)
            class_samples: list[Sample] = []
            for subject_id in subject_ids:
                class_samples.append(rng.choice(by_class_subject[class_name][subject_id]))
            selected[class_name] = class_samples
            used_subjects.update(subject_ids)
        if not failed:
            return selected
    raise ValueError(
        "could not draw a patient-disjoint episode after "
        f"{max_attempts} attempts for classes {list(class_names)}"
    )


def _episode_record(
    *,
    partition: str,
    seed: int,
    episode_index: int,
    rng: StableRNG,
    class_names: Sequence[str],
    pools: Mapping[str, Sequence[Sample]],
    shots: Sequence[int],
    queries_per_class: int,
) -> dict[str, Any]:
    ways = len(class_names)
    ordered_classes = list(class_names)
    rng.shuffle(ordered_classes)
    max_shot = max(shots)
    selected = _select_episode_samples(
        rng=rng,
        class_names=ordered_classes,
        pools=pools,
        count_per_class=max_shot + queries_per_class,
    )

    class_records = [
        {"episode_label": index, "class_name": class_name}
        for index, class_name in enumerate(ordered_classes)
    ]
    support: list[dict[str, Any]] = []
    query: list[dict[str, Any]] = []
    for episode_label, class_name in enumerate(ordered_classes):
        samples = selected[class_name]
        for shot_rank, item in enumerate(samples[:max_shot], start=1):
            support.append(
                {
                    "episode_label": episode_label,
                    "class_name": class_name,
                    "shot_rank": shot_rank,
                    **item.episode_item(),
                }
            )
        for query_index, item in enumerate(samples[max_shot:], start=1):
            query.append(
                {
                    "episode_label": episode_label,
                    "class_name": class_name,
                    "query_index": query_index,
                    **item.episode_item(),
                }
            )

    return {
        "episode_uid": f"{partition}-seed{seed}-episode{episode_index:04d}",
        "partition": partition,
        "seed": seed,
        "episode_index": episode_index,
        "ways": ways,
        "available_shots": list(shots),
        "queries_per_class": queries_per_class,
        "classes": class_records,
        "support": support,
        "query": query,
    }


def _write_episodes(
    output_dir: Path,
    config: Mapping[str, Any],
    pools: Mapping[str, Mapping[str, Sequence[Sample]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, int]]]]:
    episode_config = config["episodes"]
    shots = [int(value) for value in episode_config["shots"]]
    queries = int(episode_config["queries_per_class"])
    episode_count = int(episode_config["episodes_per_seed"])
    ways = int(episode_config["ways"])
    index: list[dict[str, Any]] = []
    pool_counts: dict[str, dict[str, dict[str, int]]] = {}
    episodes_root = output_dir / "episodes"
    episodes_root.mkdir()

    minimum_patients = max(shots) + queries
    deficits: list[str] = []
    for partition in episode_config["partitions"]:
        for class_name in config["class_partitions"][partition]:
            patient_count = len({item.subject_id for item in pools[partition][class_name]})
            if patient_count < minimum_patients:
                deficits.append(
                    f"{partition}/{class_name}: {patient_count} patients "
                    f"(requires {minimum_patients})"
                )
    if deficits:
        raise ValueError(
            "insufficient eligible patients for the frozen episode design:\n- "
            + "\n- ".join(deficits)
        )

    for partition in episode_config["partitions"]:
        partition_dir = episodes_root / partition
        partition_dir.mkdir()
        class_names = config["class_partitions"][partition]
        pool_counts[partition] = {}
        for class_name in class_names:
            class_samples = pools[partition][class_name]
            pool_counts[partition][class_name] = {
                "studies": len(class_samples),
                "patients": len({item.subject_id for item in class_samples}),
            }

        for seed in episode_config["seeds"]:
            rng = StableRNG(f"{config['protocol_id']}|{partition}|{seed}")
            file_path = partition_dir / f"seed_{int(seed):03d}.jsonl"
            with file_path.open("w", encoding="utf-8", newline="") as handle:
                for episode_index in range(episode_count):
                    if len(class_names) == ways:
                        episode_classes = list(class_names)
                    else:
                        episode_classes = rng.sample(class_names, ways)
                    record = _episode_record(
                        partition=partition,
                        seed=int(seed),
                        episode_index=episode_index,
                        rng=rng,
                        class_names=episode_classes,
                        pools=pools[partition],
                        shots=shots,
                        queries_per_class=queries,
                    )
                    handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            index.append(
                {
                    "partition": partition,
                    "seed": int(seed),
                    "episodes": episode_count,
                    "relative_path": file_path.relative_to(output_dir).as_posix(),
                    "sha256": _sha256(file_path),
                }
            )
    return index, pool_counts


def _write_episode_index(output_dir: Path, index: Sequence[Mapping[str, Any]]) -> Path:
    path = output_dir / "episodes" / "index.csv"
    fields = ["partition", "seed", "episodes", "relative_path", "sha256"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(index)
    return path


def build_protocol(
    *,
    data_root: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    check_images: bool = True,
    validate: bool = True,
) -> dict[str, Any]:
    """Build the complete protocol in a new, empty output directory."""

    root = Path(data_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    config_source = Path(config_path).expanduser().resolve()
    config = load_config(config_source)
    if not root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {root}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    source_paths = {
        name: root / relative_path for name, relative_path in config["source_files"].items()
    }
    for name, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} source file does not exist: {path}")

    missing_value = int(config["sample_policy"]["missing_label_value"])
    labels = config["labels"]
    label_rows = _load_labels(source_paths["labels"], labels, missing_value)
    split_by_dicom, subject_splits = _load_official_splits(source_paths["official_split"])
    canonical = _load_canonical_images(
        source_paths["metadata"],
        split_by_dicom,
        config["sample_policy"]["views"],
        config["sample_policy"]["view_preference"],
    )
    manifest_rows, exclusions = _construct_manifest(
        data_root=root,
        labels=labels,
        label_rows=label_rows,
        canonical_images=canonical,
        check_images=check_images,
    )

    config_copy = destination / "protocol_config.json"
    _write_json(config_copy, config)
    all_manifest, protocol_manifest = _write_manifests(
        destination, manifest_rows, labels, config
    )
    pools = _samples_by_partition_and_class(manifest_rows, config)
    episode_index, pool_counts = _write_episodes(destination, config, pools)
    index_path = _write_episode_index(destination, episode_index)

    eligible_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in manifest_rows:
        if row["eligible"] == "1":
            eligible_counts[row["official_split"]][row["class_name"]] += 1
    summary = {
        "protocol_id": config["protocol_id"],
        "canonical_frontal_studies": len(manifest_rows),
        "eligible_studies": sum(row["eligible"] == "1" for row in manifest_rows),
        "protocol_pool_studies": sum(
            len(class_samples)
            for partition_pools in pools.values()
            for class_samples in partition_pools.values()
        ),
        "official_split_patients": dict(sorted(Counter(subject_splits.values()).items())),
        "exclusions": dict(sorted(exclusions.items())),
        "eligible_studies_by_official_split_and_class": {
            split: dict(sorted(counts.items())) for split, counts in sorted(eligible_counts.items())
        },
        "episode_pool_counts": pool_counts,
        "image_existence_checked": check_images,
    }
    summary_path = destination / "build_summary.json"
    _write_json(summary_path, summary)

    artifacts = [config_copy, all_manifest, protocol_manifest, index_path, summary_path]
    lock = {
        "protocol_id": config["protocol_id"],
        "protocol_version": config["protocol_version"],
        "config_sha256": _sha256(config_copy),
        "sources": {
            name: {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in sorted(source_paths.items())
        },
        "artifacts": {
            path.relative_to(destination).as_posix(): _sha256(path) for path in artifacts
        },
        "episode_files": episode_index,
    }
    lock_path = destination / "protocol.lock.json"
    _write_json(lock_path, lock)

    if validate:
        validation_summary = validate_protocol(destination)
        _write_json(destination / "validation_summary.json", validation_summary)
    return summary


def _read_protocol_samples(
    path: Path, config: Mapping[str, Any]
) -> dict[str, Sample]:
    samples: dict[str, Sample] = {}
    subject_splits: dict[str, str] = {}
    class_to_partition = {
        class_name: partition
        for partition, class_names in config["class_partitions"].items()
        for class_name in class_names
    }
    official = config["official_split_for_partition"]
    with _open_text(path) as handle:
        reader = csv.DictReader(handle)
        required = [
            "dicom_id",
            "study_id",
            "subject_id",
            "official_split",
            "view",
            "relative_path",
            "class_name",
            "protocol_partition",
        ]
        _require_columns(reader, required, path)
        for row in reader:
            sample = Sample(
                dicom_id=row["dicom_id"],
                study_id=row["study_id"],
                subject_id=row["subject_id"],
                official_split=row["official_split"],
                view=row["view"],
                relative_path=row["relative_path"],
                class_name=row["class_name"],
            )
            expected_partition = class_to_partition.get(sample.class_name)
            if expected_partition is None:
                raise ValueError(f"unknown protocol class {sample.class_name!r}")
            if row["protocol_partition"] != expected_partition:
                raise ValueError(
                    f"sample {sample.dicom_id} has an incorrect protocol partition"
                )
            if sample.official_split != official[expected_partition]:
                raise ValueError(f"sample {sample.dicom_id} comes from the wrong official split")
            previous_split = subject_splits.setdefault(
                sample.subject_id, sample.official_split
            )
            if previous_split != sample.official_split:
                raise ValueError(
                    f"protocol samples are not patient-disjoint: subject "
                    f"{sample.subject_id} occurs in multiple official splits"
                )
            if sample.dicom_id in samples:
                raise ValueError(f"duplicate eligible DICOM ID {sample.dicom_id}")
            samples[sample.dicom_id] = sample
    return samples


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error


def support_for_shot(episode: Mapping[str, Any], shots: int) -> list[dict[str, Any]]:
    """Return the nested support prefix for a requested 1/3/5-shot setting."""

    if shots not in episode["available_shots"]:
        raise ValueError(f"shot setting {shots} is unavailable")
    return [item for item in episode["support"] if int(item["shot_rank"]) <= shots]


def _validate_episode(
    record: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    samples: Mapping[str, Sample],
    expected_partition: str,
    expected_seed: int,
    expected_index: int,
) -> None:
    episode_config = config["episodes"]
    ways = int(episode_config["ways"])
    shots = episode_config["shots"]
    max_shot = max(shots)
    query_count = int(episode_config["queries_per_class"])
    allowed_classes = set(config["class_partitions"][expected_partition])
    official_split = config["official_split_for_partition"][expected_partition]

    if record.get("partition") != expected_partition:
        raise ValueError("episode partition does not match its index entry")
    if int(record.get("seed", -1)) != expected_seed:
        raise ValueError("episode seed does not match its index entry")
    if int(record.get("episode_index", -1)) != expected_index:
        raise ValueError("episode_index is not consecutive")
    if int(record.get("ways", -1)) != ways or record.get("available_shots") != shots:
        raise ValueError("episode ways or shot settings differ from protocol config")
    if int(record.get("queries_per_class", -1)) != query_count:
        raise ValueError("episode query count differs from protocol config")

    classes = record.get("classes", [])
    class_names = [entry["class_name"] for entry in classes]
    class_labels = [int(entry["episode_label"]) for entry in classes]
    if len(classes) != ways or len(set(class_names)) != ways:
        raise ValueError("episode does not contain the configured number of unique classes")
    if not set(class_names).issubset(allowed_classes) or class_labels != list(range(ways)):
        raise ValueError("episode class names or labels are invalid")

    support = record.get("support", [])
    query = record.get("query", [])
    if len(support) != ways * max_shot or len(query) != ways * query_count:
        raise ValueError("episode has an incorrect support or query size")
    seen_dicoms: set[str] = set()
    seen_studies: set[str] = set()
    seen_subjects: set[str] = set()
    for role, items in (("support", support), ("query", query)):
        role_counts: Counter[str] = Counter()
        role_positions: dict[str, set[int]] = defaultdict(set)
        position_name = "shot_rank" if role == "support" else "query_index"
        expected_positions = set(range(1, max_shot + 1 if role == "support" else query_count + 1))
        for item in items:
            class_name = item["class_name"]
            episode_label = int(item["episode_label"])
            if episode_label >= ways or class_names[episode_label] != class_name:
                raise ValueError("episode item has an inconsistent class mapping")
            dicom_id = item["dicom_id"]
            sample = samples.get(dicom_id)
            if sample is None:
                raise ValueError(f"episode references ineligible DICOM {dicom_id}")
            for field in ("study_id", "subject_id", "relative_path", "view"):
                if item[field] != getattr(sample, field):
                    raise ValueError(f"episode item {dicom_id} has incorrect {field}")
            if sample.class_name != class_name or sample.official_split != official_split:
                raise ValueError(f"episode item {dicom_id} comes from the wrong pool")
            if dicom_id in seen_dicoms or sample.study_id in seen_studies:
                raise ValueError("support/query images or studies overlap within an episode")
            if sample.subject_id in seen_subjects:
                raise ValueError("support/query patients overlap within an episode")
            seen_dicoms.add(dicom_id)
            seen_studies.add(sample.study_id)
            seen_subjects.add(sample.subject_id)
            role_counts[class_name] += 1
            role_positions[class_name].add(int(item[position_name]))
        expected_count = max_shot if role == "support" else query_count
        if any(role_counts[name] != expected_count for name in class_names):
            raise ValueError(f"incorrect per-class {role} count")
        if any(role_positions[name] != expected_positions for name in class_names):
            raise ValueError(f"incorrect per-class {role} positions")

    for shot in shots:
        nested = support_for_shot(record, int(shot))
        counts = Counter(item["class_name"] for item in nested)
        if any(counts[name] != shot for name in class_names):
            raise ValueError(f"support prefix is not properly nested at {shot}-shot")


def validate_protocol(protocol_dir: str | Path) -> dict[str, Any]:
    """Validate artifact hashes and every saved episode."""

    root = Path(protocol_dir).expanduser().resolve()
    lock_path = root / "protocol.lock.json"
    with lock_path.open("r", encoding="utf-8") as handle:
        lock = json.load(handle)
    for relative_path, expected_hash in lock.get("artifacts", {}).items():
        artifact_path = root / relative_path
        if _sha256(artifact_path) != expected_hash:
            raise ValueError(f"artifact checksum mismatch: {artifact_path}")

    config = load_config(root / "protocol_config.json")
    if _sha256(root / "protocol_config.json") != lock.get("config_sha256"):
        raise ValueError("protocol config checksum differs from protocol.lock.json")
    samples = _read_protocol_samples(root / "protocol_samples.csv.gz", config)
    index_path = root / "episodes" / "index.csv"
    validated_files = 0
    validated_episodes = 0
    seen_uids: set[str] = set()

    with index_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(
            reader, ["partition", "seed", "episodes", "relative_path", "sha256"], index_path
        )
        index_rows = list(reader)

    locked_episode_files = lock.get("episode_files")
    normalized_index_rows = [
        {
            "partition": row["partition"],
            "seed": int(row["seed"]),
            "episodes": int(row["episodes"]),
            "relative_path": row["relative_path"],
            "sha256": row["sha256"],
        }
        for row in index_rows
    ]
    if normalized_index_rows != locked_episode_files:
        raise ValueError("episode index differs from protocol.lock.json")

    expected_files = len(REQUIRED_PARTITIONS) * len(config["episodes"]["seeds"])
    if len(index_rows) != expected_files:
        raise ValueError(f"episode index contains {len(index_rows)} files; expected {expected_files}")
    for entry in index_rows:
        path = root / entry["relative_path"]
        if _sha256(path) != entry["sha256"]:
            raise ValueError(f"episode checksum mismatch: {path}")
        expected_count = int(entry["episodes"])
        actual_count = 0
        for episode_index, record in enumerate(_iter_jsonl(path)):
            _validate_episode(
                record,
                config=config,
                samples=samples,
                expected_partition=entry["partition"],
                expected_seed=int(entry["seed"]),
                expected_index=episode_index,
            )
            uid = record["episode_uid"]
            if uid in seen_uids:
                raise ValueError(f"duplicate episode UID: {uid}")
            seen_uids.add(uid)
            actual_count += 1
        if actual_count != expected_count:
            raise ValueError(f"{path} contains {actual_count} episodes; expected {expected_count}")
        validated_files += 1
        validated_episodes += actual_count

    return {
        "status": "valid",
        "protocol_id": config["protocol_id"],
        "protocol_samples": len(samples),
        "episode_files": validated_files,
        "episodes": validated_episodes,
    }


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "mimic_cxr_protocol_v1.json"


def build_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build fixed patient-disjoint MIMIC-CXR few-shot episodes."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="do not verify that every eligible JPG exists (useful only for dry runs)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="skip the final full episode and checksum validation",
    )
    args = parser.parse_args(argv)
    summary = build_protocol(
        data_root=args.data_root,
        output_dir=args.output_dir,
        config_path=args.config,
        check_images=not args.skip_image_check,
        validate=not args.skip_validation,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def validate_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate a built MIMIC-CXR protocol.")
    parser.add_argument("protocol_dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(validate_protocol(args.protocol_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    build_main()
