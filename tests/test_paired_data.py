from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from pcb_defect.constants import CLASSES
from pcb_defect.data_prep.paired import (
    discover_converted_samples,
    render_runtime_datasets,
    write_protocol_artifacts,
)
from pcb_defect.paired_protocol import PairedProtocolConfig, build_paired_protocol


def _dataset(root: Path) -> Path:
    for board in ("01", "04", "05", "06", "07", "08", "09", "10", "11", "12"):
        count = 4 if board in {"01", "08"} else 2
        for cls_id, class_name in enumerate(CLASSES):
            for index in range(1, count + 1):
                stem = f"{board}_{class_name}_{index:02d}"
                image = root / "images" / "source" / f"{stem}.jpg"
                label = root / "labels" / "source" / f"{stem}.txt"
                image.parent.mkdir(parents=True, exist_ok=True)
                label.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(f"image:{stem}".encode())
                label.write_text(f"{cls_id} 0.5 0.5 0.1 0.1\n", encoding="ascii")
    return root


def _protocol(dataset: Path):
    samples = discover_converted_samples(dataset)
    config = PairedProtocolConfig(
        final_board="08",
        validation_board="01",
        final_per_class=2,
        exposure_per_class=2,
        validation_per_class=2,
        calibration_per_class=2,
        protocol_seed=42,
        expected_boards=10,
    )
    return build_paired_protocol(samples, config)


def test_discovery_is_sorted_content_addressed_and_path_independent(tmp_path: Path) -> None:
    left = _dataset(tmp_path / "left")
    right = _dataset(tmp_path / "elsewhere" / "right")

    left_samples = discover_converted_samples(left)
    right_samples = discover_converted_samples(right)

    assert left_samples == right_samples
    assert [sample.stem for sample in left_samples] == sorted(
        sample.stem for sample in left_samples
    )
    assert all(len(sample.image_sha256) == 64 for sample in left_samples)
    assert all(len(sample.label_sha256) == 64 for sample in left_samples)


def test_artifacts_have_verifiable_sha256_sidecars(tmp_path: Path) -> None:
    protocol = _protocol(_dataset(tmp_path / "source"))
    out = tmp_path / "reports" / "protocol"

    write_protocol_artifacts(protocol, out)

    manifest_path = out / "paired_split_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest["manifest_sha256"] == protocol.manifest_sha256
    assert manifest_bytes.endswith(b"\n")
    assert (out / "paired_split_manifest.sha256").read_text(encoding="ascii") == (
        f"{hashlib.sha256(manifest_bytes).hexdigest()}  paired_split_manifest.json\n"
    )
    assert (out / "dataset_fingerprint.sha256").read_text(encoding="ascii") == (
        f"{protocol.dataset_sha256}  normalized-sample-records\n"
    )


def test_runtime_lists_reference_source_without_copying_images(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "source")
    protocol = _protocol(dataset)
    runtime = tmp_path / "runtime"

    render_runtime_datasets(dataset, protocol, runtime)

    grouped = yaml.safe_load((runtime / "grouped" / "data.yaml").read_text(encoding="utf-8"))
    leaky = yaml.safe_load((runtime / "leaky_control" / "data.yaml").read_text(encoding="utf-8"))
    grouped_train = (runtime / "grouped" / "train.txt").read_text(encoding="utf-8").splitlines()
    leaky_train = (runtime / "leaky_control" / "train.txt").read_text(encoding="utf-8").splitlines()

    assert grouped["train"] == str((runtime / "grouped" / "train.txt").resolve())
    assert Path(leaky["test"]).read_bytes() == Path(grouped["test"]).read_bytes()
    assert len(grouped_train) == len(leaky_train)
    assert grouped_train != leaky_train
    assert not list(runtime.rglob("*.jpg"))
