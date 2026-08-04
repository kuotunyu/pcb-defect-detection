"""Calibration-only ONNX export, fidelity, and standalone runtime parity gate."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from pcb_defect.constants import CLASSES
from pcb_defect.data_prep.paired import (
    _load_spec,
    _verify_frozen_hashes,
    discover_converted_samples,
    render_runtime_datasets,
)
from pcb_defect.experiment import (
    InputLock,
    _environment,
    _git_provenance,
    _sha256_file,
    run_is_complete,
)
from pcb_defect.final_evaluation import final_evaluation_is_complete
from pcb_defect.paired_protocol import PairedProtocolConfig, build_paired_protocol
from pcb_defect.runtime_contract import configure_hermetic_ultralytics, onnxruntime_state
from pcb_defect.viz import boxes_from_ultralytics, greedy_match, iou

configure_hermetic_ultralytics()


class DeploymentError(RuntimeError):
    """The export cannot be promoted because a prerequisite or gate failed."""


def gate_passes(report: dict[str, Any]) -> bool:
    """Evaluate the two independent release gates without exception-based acceptance."""
    fidelity = report["fidelity"]
    return (
        abs(float(fidelity["delta_map50"])) <= float(fidelity["threshold"])
        and abs(float(fidelity["delta_map50_95"])) <= float(fidelity["threshold"])
        and parity_passes(report["parity"])
    )


def parity_passes(parity: dict[str, Any]) -> bool:
    """Evaluate the same-ONNX runtime parity gate."""
    return (
        int(parity["n_images"]) == int(parity["required_images"])
        and int(parity["n_failed"]) == 0
        and float(parity["min_iou"]) >= float(parity["required_min_iou"])
        and float(parity["max_conf_delta"]) <= float(parity["allowed_max_conf_delta"])
    )


def verified_deployment_selection(final_dir: Path) -> dict[str, Any]:
    """Return the selection only when it is anchored in hash-verified final metrics."""
    if not final_evaluation_is_complete(final_dir):
        raise DeploymentError("final evaluation is not complete or its metrics hash mismatches")
    try:
        final_metrics = json.loads((final_dir / "final_metrics.json").read_text(encoding="utf-8"))
        selection = final_metrics["deployment_selection"]
        sidecar = json.loads((final_dir / "deployment_selection.json").read_text(encoding="utf-8"))
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DeploymentError("deployment selection evidence is missing or invalid") from exc
    if sidecar != selection:
        raise DeploymentError("deployment selection sidecar differs from final_metrics.json")
    if (
        not isinstance(selection, dict)
        or selection.get("arm") != "grouped"
        or not selection.get("selected_before_final_test")
        or not isinstance(selection.get("seed"), int)
    ):
        raise DeploymentError("deployment checkpoint was not frozen from grouped validation")
    return selection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args(argv)
    export_and_gate(args.repo.resolve(), args.dataset.resolve(), args.workspace.resolve())
    return 0


def _write_calibration_yaml(calibration_yaml: Path, calibration_list: Path) -> None:
    calibration_yaml.write_text(
        yaml.safe_dump(
            {
                "train": str(calibration_list.resolve()),
                "val": str(calibration_list.resolve()),
                "test": str(calibration_list.resolve()),
                "names": dict(enumerate(CLASSES)),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
        newline="\n",
    )


def export_and_gate(repo: Path, dataset: Path, workspace: Path) -> None:
    started_at_utc = _utc_now()
    deployment_dir = workspace / "deployment"
    report_path = deployment_dir / "deployment_gate.json"
    selection = verified_deployment_selection(workspace / "final")
    lock = InputLock(**json.loads((workspace / "inputs" / "input_lock.json").read_text()))
    git_sha, git_dirty = _git_provenance(repo)
    if git_dirty or git_sha != lock.git_sha:
        raise DeploymentError("Git state does not match the experiment input lock")
    run_dir = workspace / "runs" / "grouped" / f"seed{selection['seed']}"
    if not run_is_complete(run_dir, lock):
        raise DeploymentError("selected grouped run is incomplete or hash-mismatched")
    source_weights = run_dir / "weights" / "best.pt"
    if report_path.is_file():
        if _deployment_gate_is_complete(
            report_path, deployment_dir, source_weights, selection, lock
        ):
            print(f"SKIP completed hash-matching deployment gate: {report_path}")
            return
        raise DeploymentError(
            f"existing deployment evidence is incomplete or mismatched: {report_path}"
        )
    if deployment_dir.exists():
        raise DeploymentError(f"partial deployment directory exists: {deployment_dir}")

    spec = _load_spec(repo / "configs" / "paired_protocol.yaml")
    protocol = build_paired_protocol(
        discover_converted_samples(dataset), PairedProtocolConfig(**spec["protocol"])
    )
    _verify_frozen_hashes(protocol, spec)
    runtime = workspace / "runtime_data"
    render_runtime_datasets(dataset, protocol, runtime)
    config_path = repo / "configs" / "deployment_gate.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    deployment_dir.mkdir(parents=True)
    calibration_yaml = deployment_dir / "calibration.yaml"
    calibration_list = runtime / "grouped" / "calibration.txt"
    calibration_paths = calibration_list.read_text(encoding="utf-8").splitlines()
    if len(calibration_paths) != config["calibration_images"]:
        raise DeploymentError(
            f"calibration set has {len(calibration_paths)} images, expected "
            f"{config['calibration_images']}"
        )
    _write_calibration_yaml(calibration_yaml, calibration_list)

    onnx_path = _export_onnx(source_weights, deployment_dir, config)
    pt_metrics = _validate_model(source_weights, calibration_yaml, config["imgsz"])
    runtime_before = onnxruntime_state()
    onnx_metrics = _validate_model(onnx_path, calibration_yaml, config["imgsz"])
    fidelity = {
        "split": "calibration",
        "threshold": config["fidelity_absolute_delta"],
        "pt": pt_metrics,
        "onnx": onnx_metrics,
        "delta_map50": onnx_metrics["map50"] - pt_metrics["map50"],
        "delta_map50_95": onnx_metrics["map50_95"] - pt_metrics["map50_95"],
    }
    parity = _standalone_parity(onnx_path, [Path(path) for path in calibration_paths], config)
    runtime_after = onnxruntime_state()
    if runtime_after != runtime_before:
        raise DeploymentError("ONNX Runtime state changed during deployment inference")
    report = {
        "schema_version": "1.0",
        "started_at_utc": started_at_utc,
        "git_sha": lock.git_sha,
        "dataset_sha256": lock.dataset_sha256,
        "manifest_sha256": lock.manifest_sha256,
        "deployment_selection": selection,
        "config_sha256": _sha256_file(config_path),
        "environment": _environment(),
        "command": list(sys.argv),
        "artifacts": {
            "source_checkpoint": str(source_weights.relative_to(workspace).as_posix()),
            "source_checkpoint_sha256": _sha256_file(source_weights),
            "onnx": "best.onnx",
            "onnx_sha256": _sha256_file(onnx_path),
        },
        "fidelity": fidelity,
        "parity": parity,
        "runtime_contract": {"before": runtime_before, "after": runtime_after},
        "completed_at_utc": _utc_now(),
    }
    report["passed"] = gate_passes(report)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    candidate_contract = {
        "schema_version": "1.0",
        "status": "passed" if report["passed"] else "blocked",
        "reason": None if report["passed"] else "Calibration fidelity or parity gate failed.",
        "filename": "best.onnx",
        "onnx_sha256": report["artifacts"]["onnx_sha256"],
        "source_checkpoint_sha256": report["artifacts"]["source_checkpoint_sha256"],
        "deployment_gate_sha256": _sha256_file(report_path),
        "hf_repo_id": None,
        "hf_revision": None,
    }
    (deployment_dir / "model_contract.candidate.json").write_text(
        json.dumps(candidate_contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if not report["passed"]:
        raise DeploymentError(f"deployment gate failed; inspect {report_path}")
    print(f"DEPLOYMENT GATE PASS: {report_path}")


def _deployment_gate_is_complete(
    report_path: Path,
    deployment_dir: Path,
    source_weights: Path,
    selection: dict[str, Any],
    lock: InputLock,
) -> bool:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        contract = json.loads(
            (deployment_dir / "model_contract.candidate.json").read_text(encoding="utf-8")
        )
        candidate = deployment_dir / "best.onnx"
        current_runtime = onnxruntime_state()
        return (
            report.get("passed") is True
            and report.get("git_sha") == lock.git_sha
            and report.get("dataset_sha256") == lock.dataset_sha256
            and report.get("manifest_sha256") == lock.manifest_sha256
            and report.get("deployment_selection") == selection
            and candidate.is_file()
            and _sha256_file(candidate) == report["artifacts"]["onnx_sha256"]
            and _sha256_file(source_weights) == report["artifacts"]["source_checkpoint_sha256"]
            and contract.get("status") == "passed"
            and contract.get("onnx_sha256") == report["artifacts"]["onnx_sha256"]
            and contract.get("source_checkpoint_sha256")
            == report["artifacts"]["source_checkpoint_sha256"]
            and contract.get("deployment_gate_sha256") == _sha256_file(report_path)
            and report["runtime_contract"]["before"]
            == report["runtime_contract"]["after"]
            == current_runtime
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _export_onnx(weights: Path, output_dir: Path, config: dict[str, Any]) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    generated = Path(
        model.export(
            format="onnx",
            imgsz=config["imgsz"],
            batch=config["batch"],
            dynamic=config["dynamic"],
            simplify=config["simplify"],
        )
    )
    destination = output_dir / "best.onnx"
    if destination.exists():
        raise DeploymentError(f"refusing to overwrite existing export: {destination}")
    shutil.copy2(generated, destination)
    return destination


def _validate_model(weights: Path, data_yaml: Path, imgsz: int) -> dict[str, float]:
    from ultralytics import YOLO

    metrics = YOLO(str(weights)).val(
        data=str(data_yaml),
        split="val",
        imgsz=imgsz,
        conf=0.001,
        iou=0.7,
        plots=False,
        verbose=False,
    )
    return {"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map)}


def _build_runtime_parity_models(
    onnx_path: Path,
    *,
    reference_factory: Callable[[str], Any] | None = None,
    standalone_factory: Callable[[Path], Any] | None = None,
) -> tuple[Any, Any]:
    if reference_factory is None:
        from ultralytics import YOLO

        reference_factory = YOLO
    if standalone_factory is None:
        from pcb_defect.e2e_onnx import OnnxYoloModel

        standalone_factory = OnnxYoloModel
    return reference_factory(str(onnx_path)), standalone_factory(onnx_path)


def _standalone_parity(
    onnx_path: Path,
    image_paths: list[Path],
    config: dict[str, Any],
) -> dict[str, Any]:
    reference_model, standalone_model = _build_runtime_parity_models(onnx_path)
    per_image = {}
    for image_path in image_paths:
        with Image.open(image_path) as image:
            reference_result = reference_model.predict(
                str(image_path),
                conf=config["parity_confidence"],
                device="cpu",
                verbose=False,
            )[0]
            reference_boxes = boxes_from_ultralytics(reference_result)
            onnx_boxes = standalone_model.predict(image, conf=config["parity_confidence"])
        match = greedy_match(onnx_boxes, reference_boxes, iou_thr=config["parity_match_iou"])
        pair_ious = [iou(onnx.xyxy, reference.xyxy) for onnx, reference in match.tp]
        conf_deltas = [abs(onnx.conf - reference.conf) for onnx, reference in match.tp]
        minimum_iou = min(pair_ious, default=1.0)
        maximum_conf_delta = max(conf_deltas, default=0.0)
        passed = (
            not match.fp
            and not match.fn
            and minimum_iou >= config["parity_min_iou"]
            and maximum_conf_delta <= config["parity_max_confidence_delta"]
        )
        per_image[image_path.stem] = {
            "n_reference": len(reference_boxes),
            "n_onnx": len(onnx_boxes),
            "n_matched": len(match.tp),
            "n_unmatched_onnx": len(match.fp),
            "n_unmatched_reference": len(match.fn),
            "min_iou": minimum_iou,
            "max_conf_delta": maximum_conf_delta,
            "passed": passed,
        }
    return {
        "split": "calibration",
        "reference_backend": "ultralytics-onnx",
        "candidate_backend": "standalone-onnxruntime",
        "onnx_sha256": _sha256_file(onnx_path),
        "n_images": len(per_image),
        "required_images": config["calibration_images"],
        "n_failed": sum(not row["passed"] for row in per_image.values()),
        "min_iou": min(row["min_iou"] for row in per_image.values()),
        "required_min_iou": config["parity_min_iou"],
        "max_conf_delta": max(row["max_conf_delta"] for row in per_image.values()),
        "allowed_max_conf_delta": config["parity_max_confidence_delta"],
        "per_image": per_image,
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
