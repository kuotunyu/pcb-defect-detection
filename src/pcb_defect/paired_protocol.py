"""Frozen paired evaluation protocol for measuring same-board split sensitivity.

The protocol deliberately holds the final-test images constant.  Its only
experimental difference is whether the training arm can see different defect
images rendered on the final-test board.  Class-wise one-for-one replacement
keeps the two training arms exactly equal in size and image-class histogram.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from pcb_defect.constants import CLASSES

PROTOCOL_VERSION = "paired-board-sensitivity-v1"


class ProtocolError(ValueError):
    """The available dataset cannot satisfy the frozen experiment contract."""


@dataclass(frozen=True)
class ProtocolSample:
    """Location-independent identity and content hashes for one dataset sample."""

    stem: str
    board_id: str
    class_name: str
    image_sha256: str
    label_sha256: str

    def as_dict(self) -> dict[str, str]:
        return {
            "stem": self.stem,
            "board_id": self.board_id,
            "class_name": self.class_name,
            "image_sha256": self.image_sha256,
            "label_sha256": self.label_sha256,
        }


@dataclass(frozen=True)
class PairedProtocolConfig:
    """Predeclared board roles and per-class partition sizes."""

    final_board: str
    validation_board: str
    final_per_class: int
    exposure_per_class: int
    validation_per_class: int
    calibration_per_class: int
    protocol_seed: int = 42
    expected_boards: int = 10
    training_seeds: tuple[int, ...] = (42, 43, 44)

    def as_dict(self) -> dict[str, Any]:
        return {
            "final_board": self.final_board,
            "validation_board": self.validation_board,
            "final_per_class": self.final_per_class,
            "exposure_per_class": self.exposure_per_class,
            "validation_per_class": self.validation_per_class,
            "calibration_per_class": self.calibration_per_class,
            "protocol_seed": self.protocol_seed,
            "expected_boards": self.expected_boards,
            "training_seeds": list(self.training_seeds),
        }


@dataclass(frozen=True)
class PairedProtocol:
    """All immutable sample assignments and their portable content digests."""

    config: PairedProtocolConfig
    samples: tuple[ProtocolSample, ...]
    final_test: tuple[str, ...]
    validation: tuple[str, ...]
    calibration: tuple[str, ...]
    grouped_train: tuple[str, ...]
    leaky_train: tuple[str, ...]
    leaky_exposure: tuple[str, ...]
    leaky_replaced: tuple[str, ...]

    @property
    def dataset_sha256(self) -> str:
        return _sha256_json([sample.as_dict() for sample in self.samples])

    def _manifest_payload(self) -> dict[str, Any]:
        by_stem = {sample.stem: sample for sample in self.samples}
        partitions = {
            "final_test": list(self.final_test),
            "validation": list(self.validation),
            "calibration": list(self.calibration),
            "grouped_train": list(self.grouped_train),
            "leaky_train": list(self.leaky_train),
            "leaky_exposure": list(self.leaky_exposure),
            "leaky_replaced": list(self.leaky_replaced),
        }
        return {
            "protocol_version": PROTOCOL_VERSION,
            "config": self.config.as_dict(),
            "dataset": {
                "sample_count": len(self.samples),
                "sha256": self.dataset_sha256,
                "samples": [sample.as_dict() for sample in self.samples],
            },
            "board_roles": {
                "final_test_and_exposure": self.config.final_board,
                "validation_and_calibration": self.config.validation_board,
                "grouped_train": sorted({by_stem[stem].board_id for stem in self.grouped_train}),
            },
            "partitions": partitions,
            "counts": {
                name: {
                    "images": len(stems),
                    "images_per_class": dict(
                        sorted(Counter(by_stem[stem].class_name for stem in stems).items())
                    ),
                }
                for name, stems in partitions.items()
            },
        }

    @property
    def manifest_sha256(self) -> str:
        return _sha256_json(self._manifest_payload())

    def to_manifest(self) -> dict[str, Any]:
        payload = self._manifest_payload()
        return {**payload, "manifest_sha256": self.manifest_sha256}


def build_paired_protocol(
    samples: list[ProtocolSample], config: PairedProtocolConfig
) -> PairedProtocol:
    """Build the protocol or fail before any training can start."""
    canonical = _validate_and_sort(samples, config)
    by_board_class: dict[tuple[str, str], list[ProtocolSample]] = defaultdict(list)
    for sample in canonical:
        by_board_class[(sample.board_id, sample.class_name)].append(sample)

    final: list[str] = []
    exposure: list[str] = []
    validation: list[str] = []
    calibration: list[str] = []

    for class_name in CLASSES:
        final_candidates = by_board_class[(config.final_board, class_name)]
        required_final = config.final_per_class + config.exposure_per_class
        if len(final_candidates) != required_final:
            raise ProtocolError(
                f"final board {config.final_board} class {class_name} must contain exactly "
                f"{required_final} images ({config.final_per_class} final + "
                f"{config.exposure_per_class} exposure), found {len(final_candidates)}"
            )
        ranked_final = _rank(final_candidates, config.protocol_seed, f"final:{class_name}")
        final.extend(sample.stem for sample in ranked_final[: config.final_per_class])
        exposure.extend(sample.stem for sample in ranked_final[config.final_per_class :])

        validation_candidates = by_board_class[(config.validation_board, class_name)]
        required_validation = config.validation_per_class + config.calibration_per_class
        if len(validation_candidates) != required_validation:
            raise ProtocolError(
                f"validation board {config.validation_board} class {class_name} must contain "
                f"exactly {required_validation} images ({config.validation_per_class} validation "
                f"+ {config.calibration_per_class} calibration), found "
                f"{len(validation_candidates)}"
            )
        ranked_validation = _rank(
            validation_candidates, config.protocol_seed, f"validation:{class_name}"
        )
        validation.extend(
            sample.stem for sample in ranked_validation[: config.validation_per_class]
        )
        calibration.extend(
            sample.stem for sample in ranked_validation[config.validation_per_class :]
        )

    grouped_samples = [
        sample
        for sample in canonical
        if sample.board_id not in {config.final_board, config.validation_board}
    ]
    grouped = {sample.stem for sample in grouped_samples}
    replacements: list[str] = []
    for class_name in CLASSES:
        class_candidates = [sample for sample in grouped_samples if sample.class_name == class_name]
        if len(class_candidates) < config.exposure_per_class:
            raise ProtocolError(
                f"grouped training class {class_name} has only {len(class_candidates)} images; "
                f"need {config.exposure_per_class} deterministic replacements"
            )
        ranked = _rank(class_candidates, config.protocol_seed, f"replace:{class_name}")
        replacements.extend(sample.stem for sample in ranked[: config.exposure_per_class])

    leaky = (grouped - set(replacements)) | set(exposure)
    protocol = PairedProtocol(
        config=config,
        samples=tuple(canonical),
        final_test=tuple(sorted(final)),
        validation=tuple(sorted(validation)),
        calibration=tuple(sorted(calibration)),
        grouped_train=tuple(sorted(grouped)),
        leaky_train=tuple(sorted(leaky)),
        leaky_exposure=tuple(sorted(exposure)),
        leaky_replaced=tuple(sorted(replacements)),
    )
    _assert_invariants(protocol)
    return protocol


def _validate_and_sort(
    samples: list[ProtocolSample], config: PairedProtocolConfig
) -> list[ProtocolSample]:
    if config.final_board == config.validation_board:
        raise ProtocolError("final and validation boards must differ")
    if not config.training_seeds:
        raise ProtocolError("at least one training seed is required")
    size_fields = (
        config.final_per_class,
        config.exposure_per_class,
        config.validation_per_class,
        config.calibration_per_class,
    )
    if any(value <= 0 for value in size_fields):
        raise ProtocolError("all per-class partition sizes must be positive")

    stems = [sample.stem for sample in samples]
    duplicate_stems = sorted(stem for stem, count in Counter(stems).items() if count != 1)
    if duplicate_stems:
        raise ProtocolError(f"duplicate sample stems: {duplicate_stems[:5]}")
    boards = sorted({sample.board_id for sample in samples})
    if len(boards) != config.expected_boards:
        raise ProtocolError(
            f"protocol requires exactly {config.expected_boards} boards, found "
            f"{len(boards)}: {boards}"
        )
    missing_roles = {config.final_board, config.validation_board} - set(boards)
    if missing_roles:
        raise ProtocolError(f"configured board roles are missing: {sorted(missing_roles)}")
    unknown_classes = sorted({sample.class_name for sample in samples} - set(CLASSES))
    if unknown_classes:
        raise ProtocolError(f"unknown image classes: {unknown_classes}")
    missing_classes = sorted(set(CLASSES) - {sample.class_name for sample in samples})
    if missing_classes:
        raise ProtocolError(f"dataset is missing image classes: {missing_classes}")
    return sorted(samples, key=lambda sample: sample.stem)


def _rank(samples: list[ProtocolSample], protocol_seed: int, purpose: str) -> list[ProtocolSample]:
    def key(sample: ProtocolSample) -> tuple[str, str]:
        token = f"{PROTOCOL_VERSION}|{protocol_seed}|{purpose}|{sample.stem}".encode()
        return hashlib.sha256(token).hexdigest(), sample.stem

    return sorted(samples, key=key)


def _assert_invariants(protocol: PairedProtocol) -> None:
    by_stem = {sample.stem: sample for sample in protocol.samples}
    final = set(protocol.final_test)
    validation = set(protocol.validation)
    calibration = set(protocol.calibration)
    grouped = set(protocol.grouped_train)
    leaky = set(protocol.leaky_train)
    exposure = set(protocol.leaky_exposure)

    if final & (validation | calibration | grouped | leaky):
        raise ProtocolError("final-test images overlap another partition")
    if validation & (calibration | grouped | leaky) or calibration & (grouped | leaky):
        raise ProtocolError("validation/calibration images overlap training")
    if any(by_stem[stem].board_id == protocol.config.final_board for stem in grouped):
        raise ProtocolError("grouped arm contains a final-board sibling")
    leaky_final_board = {
        stem for stem in leaky if by_stem[stem].board_id == protocol.config.final_board
    }
    if leaky_final_board != exposure:
        raise ProtocolError("leaky arm contains final-board samples outside frozen exposure set")
    grouped_hist = Counter(by_stem[stem].class_name for stem in grouped)
    leaky_hist = Counter(by_stem[stem].class_name for stem in leaky)
    if len(grouped) != len(leaky) or grouped_hist != leaky_hist:
        raise ProtocolError("paired training arms differ in size or image-class histogram")


def _sha256_json(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
