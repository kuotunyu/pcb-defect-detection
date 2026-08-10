from __future__ import annotations

import copy

import pytest

from pcb_defect.prediction_parity import (
    ParityConfig,
    ParityError,
    ParityThresholds,
    compare_backend_predictions,
    load_parity_config,
    prediction_parity_is_complete,
    prediction_parity_passes,
)
from pcb_defect.viz import Box


def _thresholds() -> ParityThresholds:
    return ParityThresholds(
        confidence=0.25,
        match_iou=0.5,
        required_min_iou=0.9,
        allowed_max_conf_delta=0.15,
    )


def test_cross_backend_report_records_per_box_matches_unmatched_and_empty_images() -> None:
    reference = {
        "detected": [Box(2, (10.0, 20.0, 30.0, 40.0), 0.80)],
        "empty": [],
    }
    candidates = {
        "onnxruntime_cuda_fp32": {
            "detected": [Box(2, (10.0, 20.0, 30.0, 40.0), 0.79)],
            "empty": [],
        },
        "tensorrt_fp16": {
            "detected": [
                Box(2, (10.0, 20.0, 30.0, 40.0), 0.78),
                Box(4, (50.0, 50.0, 60.0, 60.0), 0.30),
            ],
            "empty": [],
        },
    }

    report = compare_backend_predictions(
        reference,
        candidates,
        reference_backend="pytorch_fp32",
        split="calibration",
        thresholds=_thresholds(),
        required_images=2,
        config_sha256="a" * 64,
    )

    onnx = report["comparisons"]["onnxruntime_cuda_fp32"]
    assert onnx["images_both_empty"] == 1
    assert onnx["images_with_reference_detections"] == 1
    assert onnx["reference_detections"] == 1
    assert onnx["candidate_detections"] == 1
    assert onnx["matched_detections"] == 1
    assert onnx["unmatched_reference_detections"] == 0
    assert onnx["unmatched_candidate_detections"] == 0
    assert onnx["min_iou"] == 1.0
    assert onnx["max_conf_delta"] == pytest.approx(0.01)
    assert onnx["per_image"]["empty"]["min_iou"] is None
    assert onnx["per_image"]["empty"]["max_conf_delta"] is None
    assert onnx["per_image"]["empty"]["passed"] is True
    match = onnx["per_image"]["detected"]["matches"][0]
    assert match == {
        "class_id": 2,
        "reference": {"xyxy": [10.0, 20.0, 30.0, 40.0], "confidence": 0.8},
        "candidate": {"xyxy": [10.0, 20.0, 30.0, 40.0], "confidence": 0.79},
        "iou": 1.0,
        "confidence_delta": pytest.approx(0.01),
    }

    tensorrt = report["comparisons"]["tensorrt_fp16"]
    assert tensorrt["unmatched_candidate_detections"] == 1
    assert tensorrt["per_image"]["detected"]["unmatched_candidate"] == [
        {"class_id": 4, "xyxy": [50.0, 50.0, 60.0, 60.0], "confidence": 0.3}
    ]
    assert tensorrt["passed"] is False
    assert report["passed"] is False
    assert prediction_parity_is_complete(report) is True
    assert prediction_parity_passes(report) is False


def test_cross_backend_report_fails_on_iou_confidence_or_class_divergence() -> None:
    reference = {"image": [Box(1, (0.0, 0.0, 10.0, 10.0), 0.80)]}
    candidates = {
        "low_iou": {"image": [Box(1, (0.0, 0.0, 8.0, 10.0), 0.80)]},
        "confidence": {"image": [Box(1, (0.0, 0.0, 10.0, 10.0), 0.60)]},
        "class": {"image": [Box(2, (0.0, 0.0, 10.0, 10.0), 0.80)]},
    }

    report = compare_backend_predictions(
        reference,
        candidates,
        reference_backend="pytorch_fp32",
        split="calibration",
        thresholds=_thresholds(),
        required_images=1,
        config_sha256="b" * 64,
    )

    assert report["comparisons"]["low_iou"]["n_failed_images"] == 1
    assert report["comparisons"]["confidence"]["n_failed_images"] == 1
    assert report["comparisons"]["class"]["unmatched_reference_detections"] == 1
    assert report["comparisons"]["class"]["unmatched_candidate_detections"] == 1
    assert report["passed"] is False


def test_cross_backend_report_rejects_missing_or_extra_image_predictions() -> None:
    reference = {"one": [], "two": []}

    with pytest.raises(ParityError, match="image stems differ"):
        compare_backend_predictions(
            reference,
            {"onnxruntime_cuda_fp32": {"one": []}},
            reference_backend="pytorch_fp32",
            split="calibration",
            thresholds=_thresholds(),
            required_images=2,
            config_sha256="c" * 64,
        )


def test_load_parity_config_freezes_backends_split_images_and_thresholds(tmp_path) -> None:
    config_path = tmp_path / "backend_parity.yaml"
    config_path.write_text(
        """schema_version: \"1.0\"
reference_backend: pytorch_fp32
candidate_backends:
  - onnxruntime_cuda_fp32
  - tensorrt_fp16
split: calibration
required_images: 60
confidence: 0.25
match_iou: 0.5
required_min_iou: 0.9
allowed_max_conf_delta: 0.15
allow_unmatched_detections: false
""",
        encoding="utf-8",
    )

    assert load_parity_config(config_path) == ParityConfig(
        reference_backend="pytorch_fp32",
        candidate_backends=("onnxruntime_cuda_fp32", "tensorrt_fp16"),
        split="calibration",
        required_images=60,
        thresholds=_thresholds(),
        allow_unmatched_detections=False,
    )


@pytest.mark.parametrize(
    "replacement",
    [
        "allow_unmatched_detections: true",
        "reference_backend: tensorrt_fp16",
        "required_min_iou: 0.49",
    ],
)
def test_load_parity_config_rejects_contract_weakening(tmp_path, replacement: str) -> None:
    payload = """schema_version: \"1.0\"
reference_backend: pytorch_fp32
candidate_backends: [onnxruntime_cuda_fp32, tensorrt_fp16]
split: calibration
required_images: 60
confidence: 0.25
match_iou: 0.5
required_min_iou: 0.9
allowed_max_conf_delta: 0.15
allow_unmatched_detections: false
"""
    if replacement.startswith("allow_unmatched"):
        payload = payload.replace("allow_unmatched_detections: false", replacement)
    elif replacement.startswith("reference_backend"):
        payload = payload.replace("reference_backend: pytorch_fp32", replacement)
    else:
        payload = payload.replace("required_min_iou: 0.9", replacement)
    config_path = tmp_path / "backend_parity.yaml"
    config_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ParityError):
        load_parity_config(config_path)


@pytest.mark.parametrize(
    "mutation",
    [
        "aggregate_count",
        "per_box_iou",
        "per_box_confidence_delta",
        "missing_image",
        "threshold",
    ],
)
def test_prediction_parity_gate_recomputes_nested_evidence(mutation: str) -> None:
    report = compare_backend_predictions(
        {
            "detected": [Box(2, (10.0, 20.0, 30.0, 40.0), 0.80)],
            "empty": [],
        },
        {
            "onnxruntime_cuda_fp32": {
                "detected": [Box(2, (10.0, 20.0, 30.0, 40.0), 0.79)],
                "empty": [],
            }
        },
        reference_backend="pytorch_fp32",
        split="calibration",
        thresholds=_thresholds(),
        required_images=2,
        config_sha256="d" * 64,
    )
    forged = copy.deepcopy(report)
    comparison = forged["comparisons"]["onnxruntime_cuda_fp32"]
    match = comparison["per_image"]["detected"]["matches"][0]
    if mutation == "aggregate_count":
        comparison["matched_detections"] = 99
    elif mutation == "per_box_iou":
        match["iou"] = 0.1
    elif mutation == "per_box_confidence_delta":
        match["confidence_delta"] = 0.0
    elif mutation == "missing_image":
        comparison["per_image"].pop("empty")
    elif mutation == "threshold":
        forged["thresholds"]["required_min_iou"] = 0.5
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    assert prediction_parity_passes(forged) is False
