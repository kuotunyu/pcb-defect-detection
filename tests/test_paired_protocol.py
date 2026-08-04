from __future__ import annotations

from collections import Counter

import pytest

from pcb_defect.constants import CLASSES
from pcb_defect.paired_protocol import (
    PairedProtocolConfig,
    ProtocolError,
    ProtocolSample,
    build_paired_protocol,
)


def _samples() -> list[ProtocolSample]:
    rows: list[ProtocolSample] = []
    for board in ("01", "04", "05", "06", "07", "08", "09", "10", "11", "12"):
        per_class = 20 if board == "01" else 10
        for class_name in CLASSES:
            for index in range(1, per_class + 1):
                stem = f"{board}_{class_name}_{index:02d}"
                rows.append(
                    ProtocolSample(
                        stem=stem,
                        board_id=board,
                        class_name=class_name,
                        image_sha256=f"image-{stem}",
                        label_sha256=f"label-{stem}",
                    )
                )
    return rows


def _config() -> PairedProtocolConfig:
    return PairedProtocolConfig(
        final_board="08",
        validation_board="01",
        final_per_class=5,
        exposure_per_class=5,
        validation_per_class=10,
        calibration_per_class=10,
        protocol_seed=42,
        expected_boards=10,
    )


def _hist(stems: tuple[str, ...], samples: list[ProtocolSample]) -> Counter[str]:
    by_stem = {sample.stem: sample.class_name for sample in samples}
    return Counter(by_stem[stem] for stem in stems)


def test_build_protocol_enforces_paired_experiment_invariants() -> None:
    samples = _samples()
    protocol = build_paired_protocol(samples, _config())

    assert len(protocol.final_test) == 30
    assert len(protocol.validation) == 60
    assert len(protocol.calibration) == 60
    assert len(protocol.leaky_exposure) == 30

    final = set(protocol.final_test)
    validation = set(protocol.validation)
    calibration = set(protocol.calibration)
    grouped = set(protocol.grouped_train)
    leaky = set(protocol.leaky_train)
    exposure = set(protocol.leaky_exposure)

    assert final.isdisjoint(validation | calibration | grouped | leaky)
    assert validation.isdisjoint(calibration | grouped | leaky)
    assert calibration.isdisjoint(grouped | leaky)
    assert all(not stem.startswith("08_") for stem in grouped)
    assert {stem for stem in leaky if stem.startswith("08_")} == exposure
    assert len(grouped) == len(leaky)
    assert _hist(protocol.grouped_train, samples) == _hist(protocol.leaky_train, samples)
    assert _hist(protocol.final_test, samples) == Counter(dict.fromkeys(CLASSES, 5))


def test_manifest_and_hash_are_stable_when_input_order_changes() -> None:
    samples = _samples()
    forward = build_paired_protocol(samples, _config())
    reverse = build_paired_protocol(list(reversed(samples)), _config())

    assert forward.to_manifest() == reverse.to_manifest()
    assert forward.manifest_sha256 == reverse.manifest_sha256
    assert forward.dataset_sha256 == reverse.dataset_sha256
    assert len(forward.manifest_sha256) == 64
    assert len(forward.dataset_sha256) == 64


def test_protocol_fails_closed_when_dataset_has_fewer_than_ten_boards() -> None:
    samples = [sample for sample in _samples() if sample.board_id != "12"]

    with pytest.raises(ProtocolError, match="exactly 10 boards"):
        build_paired_protocol(samples, _config())


def test_protocol_fails_closed_when_final_board_lacks_siblings() -> None:
    samples = [
        sample
        for sample in _samples()
        if not (
            sample.board_id == "08"
            and sample.class_name == CLASSES[0]
            and int(sample.stem.rsplit("_", 1)[1]) > 9
        )
    ]

    with pytest.raises(ProtocolError, match="final board 08"):
        build_paired_protocol(samples, _config())


def test_config_rejects_same_final_and_validation_board() -> None:
    config = _config()
    config = PairedProtocolConfig(**{**config.as_dict(), "validation_board": "08"})

    with pytest.raises(ProtocolError, match="must differ"):
        build_paired_protocol(_samples(), config)
