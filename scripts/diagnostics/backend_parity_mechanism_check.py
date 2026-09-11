"""Mechanism check for the L4 backend-parity failure (read-only diagnostic).

Runs one PyTorch checkpoint through Ultralytics ``predict()`` on the frozen calibration images in
two input geometries and compares both against the standalone ONNX Runtime path:

* ``A``: default ``predict()`` -> ``rect=True`` -> ``LetterBox(auto=True)`` -> 1x3x352x640 input
* ``B``: ``predict(rect=False)`` -> ``LetterBox(auto=False)`` -> 1x3x640x640 input
* ``C``: ``pcb_defect.e2e_onnx.OnnxYoloModel`` (fixed 640x640, CPU provider)
* ``D``: ``pcb_defect.benchmark._ultralytics_boxes`` (the runner helper as currently committed)

The repository's frozen parity evaluator then scores ``A`` vs ``C`` (what the L4 runner compared),
``A`` vs ``B`` (same weights, geometry only), ``B`` vs ``C`` (same geometry), and ``D`` vs ``C``.
Per-image output is pseudonymized and free of coordinates and raw confidences, matching the
public-evidence policy of ``pcb_defect.l4_evidence``.

This is a mechanism demonstration for whichever checkpoint/ONNX pair is supplied; it is not a
substitute for re-running the L4 gate on the recorded checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")

from PIL import Image  # noqa: E402

SUMMARY_FIELDS = (
    "reference_detections",
    "candidate_detections",
    "matched_detections",
    "unmatched_reference_detections",
    "unmatched_candidate_detections",
    "min_iou",
    "max_conf_delta",
    "n_failed_images",
    "n_images",
    "passed",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(report: dict[str, Any], candidate: str) -> dict[str, Any]:
    comparison = report["comparisons"][candidate]
    return {field: comparison[field] for field in SUMMARY_FIELDS}


def _pseudonymized(report: dict[str, Any]) -> dict[str, Any]:
    first = next(iter(report["comparisons"]))
    order = sorted(report["comparisons"][first]["per_image"])
    image_ids = {stem: f"image_{index:03d}" for index, stem in enumerate(order, start=1)}
    sanitized: dict[str, Any] = {}
    for candidate, comparison in report["comparisons"].items():
        sanitized[candidate] = {
            image_ids[stem]: {
                "ref": row["reference_detections"],
                "cand": row["candidate_detections"],
                "matched": row["matched_detections"],
                "min_iou": row["min_iou"],
                "max_conf_delta": row["max_conf_delta"],
                "passed": row["passed"],
            }
            for stem, row in comparison["per_image"].items()
        }
    return sanitized


def _record_input_shapes(model: Any, shapes: set[tuple[int, ...]]) -> None:
    """Wrap the live predictor so the tensor shape fed to the network is observed, not assumed."""
    original = model.predictor.preprocess

    def preprocess(images: Any) -> Any:
        tensor = original(images)
        shapes.add(tuple(tensor.shape))
        return tensor

    model.predictor.preprocess = preprocess


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--pt", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    import torch
    import ultralytics
    from ultralytics import YOLO

    import pcb_defect.benchmark as benchmark_module
    from pcb_defect.e2e_onnx import OnnxYoloModel
    from pcb_defect.prediction_parity import ParityThresholds, compare_backend_predictions
    from pcb_defect.viz import boxes_from_ultralytics

    manifest = json.loads(
        (args.repo / "reports/protocol/paired_split_manifest.json").read_text("utf-8")
    )
    by_stem = {row["stem"]: row for row in manifest["dataset"]["samples"]}
    stems = manifest["partitions"]["calibration"]
    if args.limit:
        stems = stems[: args.limit]
    local = {
        path.stem: path for path in (args.repo / "data/pcb/images").rglob("*") if path.is_file()
    }
    paths = [local[stem] for stem in stems]
    for path in paths:
        if _sha256(path) != by_stem[path.stem]["image_sha256"]:
            raise SystemExit(f"calibration image hash mismatch: {path.stem}")
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())

    confidence = 0.25
    pt_rect = YOLO(str(args.pt))
    pt_square = YOLO(str(args.pt))
    pt_helper = YOLO(str(args.pt))
    pt_helper.overrides["device"] = args.device
    ort_model = OnnxYoloModel(args.onnx, providers=["CPUExecutionProvider"])

    shapes: dict[str, set[tuple[int, ...]]] = {
        "A_pt_rect_default": set(),
        "B_pt_rect_false": set(),
        "D_pt_benchmark_helper": set(),
    }
    # The first call creates each predictor; then observe every subsequent input tensor shape.
    pt_rect.predict(images[0], conf=confidence, verbose=False, device=args.device)
    _record_input_shapes(pt_rect, shapes["A_pt_rect_default"])
    pt_square.predict(
        images[0], conf=confidence, verbose=False, device=args.device, imgsz=640, rect=False
    )
    _record_input_shapes(pt_square, shapes["B_pt_rect_false"])
    benchmark_module._ultralytics_boxes(pt_helper, images[0], confidence)
    _record_input_shapes(pt_helper, shapes["D_pt_benchmark_helper"])

    predictions: dict[str, dict[str, Any]] = {
        "A_pt_rect_default": {},
        "B_pt_rect_false": {},
        "C_ort_standalone_640": {},
        "D_pt_benchmark_helper": {},
    }
    started = time.perf_counter()
    for path, image in zip(paths, images, strict=True):
        result_a = pt_rect.predict(image, conf=confidence, verbose=False, device=args.device)[0]
        result_b = pt_square.predict(
            image, conf=confidence, verbose=False, device=args.device, imgsz=640, rect=False
        )[0]
        predictions["A_pt_rect_default"][path.stem] = boxes_from_ultralytics(result_a)
        predictions["B_pt_rect_false"][path.stem] = boxes_from_ultralytics(result_b)
        predictions["C_ort_standalone_640"][path.stem] = ort_model.predict(image, conf=confidence)
        predictions["D_pt_benchmark_helper"][path.stem] = benchmark_module._ultralytics_boxes(
            pt_helper, image, confidence
        )

    thresholds = ParityThresholds(
        confidence=confidence, match_iou=0.5, required_min_iou=0.9, allowed_max_conf_delta=0.15
    )

    def compare(reference: str, candidates: list[str]) -> dict[str, Any]:
        return compare_backend_predictions(
            predictions[reference],
            {candidate: predictions[candidate] for candidate in candidates},
            reference_backend=reference,
            split="calibration",
            thresholds=thresholds,
            required_images=len(stems),
            config_sha256="0" * 64,
        )

    report_a = compare("A_pt_rect_default", ["B_pt_rect_false", "C_ort_standalone_640"])
    report_b = compare("B_pt_rect_false", ["C_ort_standalone_640"])
    report_d = compare("D_pt_benchmark_helper", ["C_ort_standalone_640", "B_pt_rect_false"])

    result = {
        "schema_version": "diagnostic-1.0",
        "purpose": "mechanism check only; local prototype model, not the L4 checkpoint",
        "environment": {
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "device_arg": args.device,
            "onnxruntime": __import__("onnxruntime").__version__,
            "python": sys.version.split()[0],
        },
        "artifacts": {
            "pt_sha256": _sha256(args.pt),
            "onnx_sha256": _sha256(args.onnx),
            "pt_bytes": args.pt.stat().st_size,
            "onnx_bytes": args.onnx.stat().st_size,
        },
        "calibration": {
            "n_images": len(stems),
            "image_size_wh": sorted({image.size for image in images}),
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "observed_input_tensor_shapes": {
            key: sorted(list(shape) for shape in value) for key, value in shapes.items()
        },
        "thresholds": report_a["thresholds"],
        "summary": {
            "A_pt_rect_default__vs__C_ort_640 (what the L4 runner compared)": _summary(
                report_a, "C_ort_standalone_640"
            ),
            "A_pt_rect_default__vs__B_pt_rect_false (same weights, geometry only)": _summary(
                report_a, "B_pt_rect_false"
            ),
            "B_pt_rect_false__vs__C_ort_640 (same geometry)": _summary(
                report_b, "C_ort_standalone_640"
            ),
            "D_patched_runner_helper__vs__C_ort_640 (what a re-run would compare)": _summary(
                report_d, "C_ort_standalone_640"
            ),
            "D_patched_runner_helper__vs__B_pt_rect_false (helper == explicit rect=False)": (
                _summary(report_d, "B_pt_rect_false")
            ),
        },
        "runner_helper_module": "src/pcb_defect/benchmark.py",
        "per_image": {
            "reference=A_pt_rect_default": _pseudonymized(report_a),
            "reference=B_pt_rect_false": _pseudonymized(report_b),
            "reference=D_pt_benchmark_helper": _pseudonymized(report_d),
        },
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"shapes": result["observed_input_tensor_shapes"], "summary": result["summary"]},
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
