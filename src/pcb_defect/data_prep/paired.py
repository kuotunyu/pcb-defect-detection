"""Portable dataset discovery and runtime files for the paired experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from pcb_defect.constants import CLASS_TO_ID, CLASSES
from pcb_defect.data_prep.split import board_id
from pcb_defect.paired_protocol import (
    PROTOCOL_VERSION,
    PairedProtocol,
    PairedProtocolConfig,
    ProtocolError,
    ProtocolSample,
    build_paired_protocol,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def main(argv: list[str] | None = None) -> int:
    """Verify the frozen protocol and generate only portable/runtime derivatives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/pcb"))
    parser.add_argument("--config", type=Path, default=Path("configs/paired_protocol.yaml"))
    parser.add_argument("--artifacts", type=Path, default=Path("reports/protocol"))
    parser.add_argument("--runtime", type=Path, default=Path("data/paired"))
    parser.add_argument(
        "--show-hashes",
        action="store_true",
        help="print discovered hashes without writing artifacts (maintainer freeze workflow)",
    )
    args = parser.parse_args(argv)

    spec = _load_spec(args.config)
    config = PairedProtocolConfig(**spec["protocol"])
    protocol = build_paired_protocol(discover_converted_samples(args.source), config)
    if args.show_hashes:
        print(f"dataset_sha256={protocol.dataset_sha256}")
        print(f"manifest_sha256={protocol.manifest_sha256}")
        return 0

    _verify_frozen_hashes(protocol, spec)
    write_protocol_artifacts(protocol, args.artifacts)
    render_runtime_datasets(args.source, protocol, args.runtime)
    print(f"protocol manifest: {(args.artifacts / 'paired_split_manifest.json').resolve()}")
    print(f"manifest_sha256={protocol.manifest_sha256}")
    print(f"dataset_sha256={protocol.dataset_sha256}")
    print(f"runtime lists: {args.runtime.resolve()}")
    return 0


def discover_converted_samples(dataset_root: Path) -> list[ProtocolSample]:
    """Read a converted YOLO dataset into location-independent sample records."""
    root = dataset_root.resolve()
    images_root = root / "images"
    labels_root = root / "labels"
    if not images_root.is_dir() or not labels_root.is_dir():
        raise ProtocolError(f"expected images/ and labels/ under {root}")

    records: list[ProtocolSample] = []
    seen: set[str] = set()
    image_paths = sorted(
        path
        for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    for image_path in image_paths:
        stem = image_path.stem
        if stem in seen:
            raise ProtocolError(f"duplicate image stem in converted dataset: {stem}")
        seen.add(stem)
        relative = image_path.relative_to(images_root)
        label_path = labels_root / relative.parent / f"{stem}.txt"
        if not label_path.is_file():
            raise ProtocolError(f"missing label for image {relative.as_posix()}")
        class_name = _class_from_stem(stem)
        _validate_label_class(label_path, class_name)
        records.append(
            ProtocolSample(
                stem=stem,
                board_id=board_id(stem),
                class_name=class_name,
                image_sha256=_sha256_file(image_path),
                label_sha256=_sha256_file(label_path),
            )
        )
    if not records:
        raise ProtocolError(f"no supported images found under {images_root}")
    return sorted(records, key=lambda sample: sample.stem)


def write_protocol_artifacts(protocol: PairedProtocol, output_dir: Path) -> None:
    """Write the canonical manifest and independently verifiable hash sidecars."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "paired_split_manifest.json"
    manifest_bytes = (
        json.dumps(protocol.to_manifest(), indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    file_hash = hashlib.sha256(manifest_bytes).hexdigest()
    (output_dir / "paired_split_manifest.sha256").write_text(
        f"{file_hash}  {manifest_path.name}\n", encoding="ascii", newline="\n"
    )
    (output_dir / "dataset_fingerprint.sha256").write_text(
        f"{protocol.dataset_sha256}  normalized-sample-records\n",
        encoding="ascii",
        newline="\n",
    )


def render_runtime_datasets(
    dataset_root: Path, protocol: PairedProtocol, output_root: Path
) -> None:
    """Create small YOLO path lists for both arms without copying dataset images."""
    actual = tuple(discover_converted_samples(dataset_root))
    if actual != protocol.samples:
        raise ProtocolError(
            "dataset content no longer matches the frozen protocol manifest; refusing to train"
        )
    image_by_stem = _image_paths_by_stem(dataset_root)
    shared = {
        "val": protocol.validation,
        "test": protocol.final_test,
        "calibration": protocol.calibration,
    }
    for arm, train_stems in (
        ("grouped", protocol.grouped_train),
        ("leaky_control", protocol.leaky_train),
    ):
        arm_dir = output_root.resolve() / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        partitions = {"train": train_stems, **shared}
        list_paths: dict[str, str] = {}
        for partition, stems in partitions.items():
            list_path = arm_dir / f"{partition}.txt"
            lines = [str(image_by_stem[stem].resolve()) for stem in stems]
            list_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
            list_paths[partition] = str(list_path.resolve())
        data_yaml = {
            "train": list_paths["train"],
            "val": list_paths["val"],
            "test": list_paths["test"],
            "names": dict(enumerate(CLASSES)),
        }
        with (arm_dir / "data.yaml").open("w", encoding="utf-8", newline="\n") as handle:
            yaml.safe_dump(data_yaml, handle, allow_unicode=True, sort_keys=False)


def _image_paths_by_stem(dataset_root: Path) -> dict[str, Path]:
    image_paths = [
        path
        for path in (dataset_root.resolve() / "images").rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    result: dict[str, Path] = {}
    for path in image_paths:
        if path.stem in result:
            raise ProtocolError(f"duplicate image stem in converted dataset: {path.stem}")
        result[path.stem] = path
    return result


def _class_from_stem(stem: str) -> str:
    board = board_id(stem)
    matches = [
        class_name
        for class_name in CLASSES
        if stem.startswith(f"{board}_{class_name}_")
        and stem.removeprefix(f"{board}_{class_name}_").isdigit()
    ]
    if len(matches) != 1:
        raise ProtocolError(f"cannot derive one canonical class from image stem {stem!r}")
    return matches[0]


def _validate_label_class(label_path: Path, class_name: str) -> None:
    lines = label_path.read_text(encoding="ascii").strip().splitlines()
    if not lines:
        raise ProtocolError(f"empty label file: {label_path}")
    expected = CLASS_TO_ID[class_name]
    try:
        observed = {int(line.split()[0]) for line in lines}
    except (IndexError, ValueError) as exc:
        raise ProtocolError(f"malformed YOLO label file: {label_path}") from exc
    if observed != {expected}:
        raise ProtocolError(
            f"label classes for {label_path.name} are {sorted(observed)}, expected only {expected}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_spec(path: Path) -> dict:
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ProtocolError(f"protocol config is not a mapping: {path}")
    if spec.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError(
            f"config protocol_version must be {PROTOCOL_VERSION!r}, "
            f"found {spec.get('protocol_version')!r}"
        )
    if not isinstance(spec.get("protocol"), dict):
        raise ProtocolError("protocol config is missing the 'protocol' mapping")
    return spec


def _verify_frozen_hashes(protocol: PairedProtocol, spec: dict) -> None:
    expected = spec.get("frozen_hashes")
    if not isinstance(expected, dict):
        raise ProtocolError("protocol config is missing frozen_hashes")
    comparisons = {
        "dataset_sha256": protocol.dataset_sha256,
        "manifest_sha256": protocol.manifest_sha256,
    }
    for name, actual in comparisons.items():
        value = expected.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ProtocolError(f"frozen hash {name} is missing or malformed")
        if value != actual:
            raise ProtocolError(f"{name} mismatch: expected {value}, found {actual}")


if __name__ == "__main__":
    raise SystemExit(main())
