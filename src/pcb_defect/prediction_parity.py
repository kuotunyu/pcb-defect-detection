"""Deterministic, per-box prediction parity reporting across inference backends."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from pcb_defect.viz import Box, greedy_match, iou

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FROZEN_THRESHOLDS = {
    "confidence": 0.25,
    "match_iou": 0.5,
    "required_min_iou": 0.9,
    "allowed_max_conf_delta": 0.15,
}


class ParityError(ValueError):
    """Raised when prediction parity inputs violate the frozen comparison contract."""


@dataclass(frozen=True, slots=True)
class ParityThresholds:
    """Frozen thresholds applied to every backend and calibration image."""

    confidence: float
    match_iou: float
    required_min_iou: float
    allowed_max_conf_delta: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ParityError(f"{name} must be a finite number")
        if not 0.0 <= self.confidence <= 1.0:
            raise ParityError("confidence must be between 0 and 1")
        if not 0.0 < self.match_iou <= 1.0:
            raise ParityError("match_iou must be in (0, 1]")
        if not self.match_iou <= self.required_min_iou <= 1.0:
            raise ParityError("required_min_iou must be between match_iou and 1")
        if not 0.0 <= self.allowed_max_conf_delta <= 1.0:
            raise ParityError("allowed_max_conf_delta must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class ParityConfig:
    """The versioned cross-backend parity protocol loaded from YAML."""

    reference_backend: str
    candidate_backends: tuple[str, ...]
    split: str
    required_images: int
    thresholds: ParityThresholds
    allow_unmatched_detections: bool


def load_parity_config(path: Path) -> ParityConfig:
    """Load and strictly validate the frozen backend-parity protocol."""
    required_fields = {
        "schema_version",
        "reference_backend",
        "candidate_backends",
        "split",
        "required_images",
        "confidence",
        "match_iou",
        "required_min_iou",
        "allowed_max_conf_delta",
        "allow_unmatched_detections",
    }
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ParityError(f"cannot read parity config: {path}") from exc
    if not isinstance(payload, dict) or set(payload) != required_fields:
        raise ParityError("parity config fields do not match schema 1.0")
    if payload["schema_version"] != "1.0":
        raise ParityError("unsupported parity config schema")
    if payload["reference_backend"] != "pytorch_fp32":
        raise ParityError("pytorch_fp32 must remain the reference backend")
    candidates = payload["candidate_backends"]
    expected_candidates = ("onnxruntime_cuda_fp32", "tensorrt_fp16")
    if not isinstance(candidates, list) or tuple(candidates) != expected_candidates:
        raise ParityError("candidate_backends must be ONNX Runtime CUDA and TensorRT FP16")
    if payload["split"] != "calibration":
        raise ParityError("prediction parity must use the calibration split")
    required_images = payload["required_images"]
    if (
        not isinstance(required_images, int)
        or isinstance(required_images, bool)
        or required_images < 1
    ):
        raise ParityError("required_images must be a positive integer")
    if payload["allow_unmatched_detections"] is not False:
        raise ParityError("unmatched detections must remain disallowed")
    try:
        thresholds = ParityThresholds(
            confidence=payload["confidence"],
            match_iou=payload["match_iou"],
            required_min_iou=payload["required_min_iou"],
            allowed_max_conf_delta=payload["allowed_max_conf_delta"],
        )
    except (KeyError, TypeError) as exc:
        raise ParityError("parity thresholds are malformed") from exc
    if asdict(thresholds) != _FROZEN_THRESHOLDS:
        raise ParityError("prediction parity thresholds are frozen by schema 1.0")
    return ParityConfig(
        reference_backend="pytorch_fp32",
        candidate_backends=expected_candidates,
        split="calibration",
        required_images=required_images,
        thresholds=thresholds,
        allow_unmatched_detections=False,
    )


def _serialize_box(box: Box) -> dict[str, Any]:
    return {
        "class_id": int(box.cls_id),
        "xyxy": [float(value) for value in box.xyxy],
        "confidence": float(box.conf),
    }


def _validate_box(box: Box, *, backend: str, stem: str) -> None:
    if not isinstance(box, Box):
        raise ParityError(f"{backend}/{stem} contains a non-Box prediction")
    values = (*box.xyxy, box.conf)
    if any(not math.isfinite(float(value)) for value in values):
        raise ParityError(f"{backend}/{stem} contains a non-finite prediction")


def _validate_predictions(
    predictions: Mapping[str, list[Box]], *, backend: str, required_images: int | None
) -> set[str]:
    stems = set(predictions)
    if required_images is not None and len(stems) != required_images:
        raise ParityError(
            f"{backend} must contain exactly {required_images} image stems; found {len(stems)}"
        )
    if any(not isinstance(stem, str) or not stem for stem in stems):
        raise ParityError(f"{backend} contains an invalid image stem")
    for stem, boxes in predictions.items():
        if not isinstance(boxes, list):
            raise ParityError(f"{backend}/{stem} predictions must be a list")
        for box in boxes:
            _validate_box(box, backend=backend, stem=stem)
    return stems


def _compare_image(
    reference: list[Box], candidate: list[Box], thresholds: ParityThresholds
) -> dict[str, Any]:
    matched = greedy_match(candidate, reference, iou_thr=thresholds.match_iou)
    matches: list[dict[str, Any]] = []
    pair_ious: list[float] = []
    confidence_deltas: list[float] = []
    for candidate_box, reference_box in matched.tp:
        pair_iou = float(iou(candidate_box.xyxy, reference_box.xyxy))
        confidence_delta = abs(float(candidate_box.conf) - float(reference_box.conf))
        pair_ious.append(pair_iou)
        confidence_deltas.append(confidence_delta)
        matches.append(
            {
                "class_id": int(reference_box.cls_id),
                "reference": {
                    "xyxy": [float(value) for value in reference_box.xyxy],
                    "confidence": float(reference_box.conf),
                },
                "candidate": {
                    "xyxy": [float(value) for value in candidate_box.xyxy],
                    "confidence": float(candidate_box.conf),
                },
                "iou": pair_iou,
                "confidence_delta": confidence_delta,
            }
        )

    min_iou = min(pair_ious) if pair_ious else None
    max_conf_delta = max(confidence_deltas) if confidence_deltas else None
    passed = (
        not matched.fp
        and not matched.fn
        and (min_iou is None or min_iou >= thresholds.required_min_iou)
        and (max_conf_delta is None or max_conf_delta <= thresholds.allowed_max_conf_delta)
    )
    return {
        "reference_detections": len(reference),
        "candidate_detections": len(candidate),
        "matched_detections": len(matched.tp),
        "unmatched_reference_detections": len(matched.fn),
        "unmatched_candidate_detections": len(matched.fp),
        "min_iou": min_iou,
        "max_conf_delta": max_conf_delta,
        "matches": matches,
        "unmatched_reference": [_serialize_box(box) for box in matched.fn],
        "unmatched_candidate": [_serialize_box(box) for box in matched.fp],
        "passed": passed,
    }


def compare_backend_predictions(
    reference: Mapping[str, list[Box]],
    candidates: Mapping[str, Mapping[str, list[Box]]],
    *,
    reference_backend: str,
    split: str,
    thresholds: ParityThresholds,
    required_images: int,
    config_sha256: str,
) -> dict[str, Any]:
    """Compare each candidate backend to one reference on identical image stems."""
    if (
        not isinstance(required_images, int)
        or isinstance(required_images, bool)
        or required_images < 1
    ):
        raise ParityError("required_images must be a positive integer")
    if not isinstance(reference_backend, str) or not reference_backend:
        raise ParityError("reference_backend must be a non-empty string")
    if not isinstance(split, str) or not split:
        raise ParityError("split must be a non-empty string")
    if not isinstance(config_sha256, str) or _SHA256_RE.fullmatch(config_sha256) is None:
        raise ParityError("config_sha256 must be a lowercase SHA-256 digest")
    if not candidates:
        raise ParityError("at least one candidate backend is required")

    reference_stems = _validate_predictions(
        reference, backend=reference_backend, required_images=required_images
    )
    comparisons: dict[str, dict[str, Any]] = {}
    for backend in sorted(candidates):
        if not isinstance(backend, str) or not backend or backend == reference_backend:
            raise ParityError("candidate backend names must be non-empty and distinct")
        predictions = candidates[backend]
        candidate_stems = _validate_predictions(predictions, backend=backend, required_images=None)
        if candidate_stems != reference_stems:
            missing = sorted(reference_stems - candidate_stems)
            extra = sorted(candidate_stems - reference_stems)
            raise ParityError(
                f"{backend} image stems differ from {reference_backend}: "
                f"missing={missing}, extra={extra}"
            )

        per_image: dict[str, dict[str, Any]] = {}
        for stem in sorted(reference_stems):
            per_image[stem] = _compare_image(reference[stem], predictions[stem], thresholds)

        matched_ious = [
            match["iou"] for image_report in per_image.values() for match in image_report["matches"]
        ]
        confidence_deltas = [
            match["confidence_delta"]
            for image_report in per_image.values()
            for match in image_report["matches"]
        ]
        n_failed_images = sum(not image_report["passed"] for image_report in per_image.values())
        comparison = {
            "n_images": required_images,
            "images_both_empty": sum(
                image_report["reference_detections"] == 0
                and image_report["candidate_detections"] == 0
                for image_report in per_image.values()
            ),
            "images_with_reference_detections": sum(
                image_report["reference_detections"] > 0 for image_report in per_image.values()
            ),
            "reference_detections": sum(
                image_report["reference_detections"] for image_report in per_image.values()
            ),
            "candidate_detections": sum(
                image_report["candidate_detections"] for image_report in per_image.values()
            ),
            "matched_detections": sum(
                image_report["matched_detections"] for image_report in per_image.values()
            ),
            "unmatched_reference_detections": sum(
                image_report["unmatched_reference_detections"]
                for image_report in per_image.values()
            ),
            "unmatched_candidate_detections": sum(
                image_report["unmatched_candidate_detections"]
                for image_report in per_image.values()
            ),
            "min_iou": min(matched_ious) if matched_ious else None,
            "max_conf_delta": max(confidence_deltas) if confidence_deltas else None,
            "n_failed_images": n_failed_images,
            "per_image": per_image,
            "passed": n_failed_images == 0,
        }
        comparisons[backend] = comparison

    return {
        "schema_version": "1.0",
        "split": split,
        "reference_backend": reference_backend,
        "candidate_backends": sorted(comparisons),
        "required_images": required_images,
        "n_images": len(reference_stems),
        "config_sha256": config_sha256,
        "thresholds": asdict(thresholds),
        "comparisons": comparisons,
        "passed": all(comparison["passed"] for comparison in comparisons.values()),
    }


def prediction_parity_is_complete(report: object) -> bool:
    """Return True when every nested value is complete and internally consistent."""
    try:
        return _prediction_parity_is_complete(report)
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return False


def prediction_parity_passes(report: object) -> bool:
    """Return True only for a structurally complete report whose frozen gate passed."""
    return (
        prediction_parity_is_complete(report)
        and isinstance(report, dict)
        and report.get("passed") is True
    )


def _prediction_parity_is_complete(report: object) -> bool:
    report_fields = {
        "schema_version",
        "split",
        "reference_backend",
        "candidate_backends",
        "required_images",
        "n_images",
        "config_sha256",
        "thresholds",
        "comparisons",
        "passed",
    }
    if not isinstance(report, dict) or set(report) != report_fields:
        return False
    required_images = report["required_images"]
    if (
        report["schema_version"] != "1.0"
        or not isinstance(report["split"], str)
        or not report["split"]
        or not isinstance(report["reference_backend"], str)
        or not report["reference_backend"]
        or not isinstance(required_images, int)
        or isinstance(required_images, bool)
        or required_images < 1
        or report["n_images"] != required_images
        or not isinstance(report["config_sha256"], str)
        or _SHA256_RE.fullmatch(report["config_sha256"]) is None
        or report["thresholds"] != _FROZEN_THRESHOLDS
    ):
        return False
    thresholds = ParityThresholds(**report["thresholds"])
    candidates = report["candidate_backends"]
    comparisons = report["comparisons"]
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(not isinstance(backend, str) or not backend for backend in candidates)
        or candidates != sorted(comparisons)
        or set(candidates) != set(comparisons)
    ):
        return False

    all_passed = True
    for backend in candidates:
        comparison = comparisons[backend]
        if not _comparison_is_consistent(comparison, required_images, thresholds):
            return False
        all_passed = all_passed and comparison["passed"]
    return report["passed"] is all_passed


def _comparison_is_consistent(
    comparison: object, required_images: int, thresholds: ParityThresholds
) -> bool:
    comparison_fields = {
        "n_images",
        "images_both_empty",
        "images_with_reference_detections",
        "reference_detections",
        "candidate_detections",
        "matched_detections",
        "unmatched_reference_detections",
        "unmatched_candidate_detections",
        "min_iou",
        "max_conf_delta",
        "n_failed_images",
        "per_image",
        "passed",
    }
    if not isinstance(comparison, dict) or set(comparison) != comparison_fields:
        return False
    per_image = comparison["per_image"]
    if (
        comparison["n_images"] != required_images
        or not isinstance(per_image, dict)
        or len(per_image) != required_images
        or any(not isinstance(stem, str) or not stem for stem in per_image)
    ):
        return False
    image_reports: list[dict[str, Any]] = []
    for image_report in per_image.values():
        if not _image_report_is_consistent(image_report, thresholds):
            return False
        image_reports.append(image_report)
    match_ious = [match["iou"] for image in image_reports for match in image["matches"]]
    confidence_deltas = [
        match["confidence_delta"] for image in image_reports for match in image["matches"]
    ]
    expected = {
        "n_images": required_images,
        "images_both_empty": sum(
            image["reference_detections"] == 0 and image["candidate_detections"] == 0
            for image in image_reports
        ),
        "images_with_reference_detections": sum(
            image["reference_detections"] > 0 for image in image_reports
        ),
        "reference_detections": sum(image["reference_detections"] for image in image_reports),
        "candidate_detections": sum(image["candidate_detections"] for image in image_reports),
        "matched_detections": sum(image["matched_detections"] for image in image_reports),
        "unmatched_reference_detections": sum(
            image["unmatched_reference_detections"] for image in image_reports
        ),
        "unmatched_candidate_detections": sum(
            image["unmatched_candidate_detections"] for image in image_reports
        ),
        "min_iou": min(match_ious) if match_ious else None,
        "max_conf_delta": max(confidence_deltas) if confidence_deltas else None,
        "n_failed_images": sum(not image["passed"] for image in image_reports),
        "per_image": per_image,
        "passed": all(image["passed"] for image in image_reports),
    }
    return comparison == expected


def _image_report_is_consistent(report: object, thresholds: ParityThresholds) -> bool:
    image_fields = {
        "reference_detections",
        "candidate_detections",
        "matched_detections",
        "unmatched_reference_detections",
        "unmatched_candidate_detections",
        "min_iou",
        "max_conf_delta",
        "matches",
        "unmatched_reference",
        "unmatched_candidate",
        "passed",
    }
    if not isinstance(report, dict) or set(report) != image_fields:
        return False
    matches = report["matches"]
    unmatched_reference = report["unmatched_reference"]
    unmatched_candidate = report["unmatched_candidate"]
    if not all(
        isinstance(values, list) for values in (matches, unmatched_reference, unmatched_candidate)
    ):
        return False
    if any(
        not _serialized_box_is_valid(box) for box in (*unmatched_reference, *unmatched_candidate)
    ):
        return False
    if any(not _serialized_match_is_valid(match) for match in matches):
        return False
    match_ious = [match["iou"] for match in matches]
    confidence_deltas = [match["confidence_delta"] for match in matches]
    min_iou = min(match_ious) if match_ious else None
    max_conf_delta = max(confidence_deltas) if confidence_deltas else None
    passed = (
        not unmatched_reference
        and not unmatched_candidate
        and (min_iou is None or min_iou >= thresholds.required_min_iou)
        and (max_conf_delta is None or max_conf_delta <= thresholds.allowed_max_conf_delta)
    )
    expected = {
        "reference_detections": len(matches) + len(unmatched_reference),
        "candidate_detections": len(matches) + len(unmatched_candidate),
        "matched_detections": len(matches),
        "unmatched_reference_detections": len(unmatched_reference),
        "unmatched_candidate_detections": len(unmatched_candidate),
        "min_iou": min_iou,
        "max_conf_delta": max_conf_delta,
        "matches": matches,
        "unmatched_reference": unmatched_reference,
        "unmatched_candidate": unmatched_candidate,
        "passed": passed,
    }
    return report == expected


def _serialized_box_is_valid(payload: object) -> bool:
    if not isinstance(payload, dict) or set(payload) != {"class_id", "xyxy", "confidence"}:
        return False
    return _serialized_detection_is_valid(payload, require_class=True)


def _serialized_match_is_valid(payload: object) -> bool:
    fields = {"class_id", "reference", "candidate", "iou", "confidence_delta"}
    if not isinstance(payload, dict) or set(payload) != fields:
        return False
    class_id = payload["class_id"]
    if not isinstance(class_id, int) or isinstance(class_id, bool) or class_id < 0:
        return False
    reference = payload["reference"]
    candidate = payload["candidate"]
    if not _serialized_detection_is_valid(reference) or not _serialized_detection_is_valid(
        candidate
    ):
        return False
    reference_box = Box(class_id, tuple(reference["xyxy"]), reference["confidence"])
    candidate_box = Box(class_id, tuple(candidate["xyxy"]), candidate["confidence"])
    expected_iou = iou(candidate_box.xyxy, reference_box.xyxy)
    expected_delta = abs(candidate_box.conf - reference_box.conf)
    return payload["iou"] == expected_iou and payload["confidence_delta"] == expected_delta


def _serialized_detection_is_valid(payload: object, *, require_class: bool = False) -> bool:
    fields = {"xyxy", "confidence"}
    if require_class:
        fields.add("class_id")
    if not isinstance(payload, dict) or set(payload) != fields:
        return False
    if require_class:
        class_id = payload["class_id"]
        if not isinstance(class_id, int) or isinstance(class_id, bool) or class_id < 0:
            return False
    xyxy = payload["xyxy"]
    confidence = payload["confidence"]
    return (
        isinstance(xyxy, list)
        and len(xyxy) == 4
        and all(_is_finite_number(value) for value in xyxy)
        and _is_finite_number(confidence)
        and 0.0 <= confidence <= 1.0
    )


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
