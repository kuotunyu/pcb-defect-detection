"""One-shot common-final-test evaluation for all six paired runs."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
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
from pcb_defect.experiment import ARMS, InputLock, _git_provenance, _sha256_file, run_is_complete
from pcb_defect.paired_protocol import PairedProtocolConfig, build_paired_protocol
from pcb_defect.viz import boxes_from_ultralytics, greedy_match, load_yolo_labels


class FinalEvaluationError(RuntimeError):
    """The final-test freeze or one-shot gate was violated."""


def paired_bootstrap_ci(
    deltas: list[float], *, n_resamples: int = 10_000, seed: int = 20_260_803
) -> dict[str, float | int | str]:
    """Bootstrap the paired mean difference with final-test image as the unit."""
    if not deltas:
        raise FinalEvaluationError("paired bootstrap requires at least one image delta")
    if n_resamples < 100:
        raise FinalEvaluationError("paired bootstrap requires at least 100 resamples")
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    return {
        "unit": "final-test-image",
        "n_images": n,
        "n_resamples": n_resamples,
        "seed": seed,
        "mean_delta": statistics.fmean(deltas),
        "ci95_low": _quantile(means, 0.025),
        "ci95_high": _quantile(means, 0.975),
    }


def choose_grouped_deployment_seed(validation: dict[int, dict[str, Any]]) -> int:
    """Pre-final selection by grouped validation mAP50-95; lower seed breaks ties."""
    if not validation:
        raise FinalEvaluationError("no grouped validation metrics are available")
    try:
        scores = {seed: float(metrics["map50_95"]) for seed, metrics in validation.items()}
    except (KeyError, TypeError, ValueError) as exc:
        raise FinalEvaluationError("grouped validation metrics lack finite map50_95") from exc
    if not all(math.isfinite(value) for value in scores.values()):
        raise FinalEvaluationError("grouped validation metrics lack finite map50_95")
    return min(scores, key=lambda seed: (-scores[seed], seed))


def begin_final_test_once(marker: Path, payload: dict[str, Any]) -> None:
    """Create the irreversible test-spend marker, refusing every second attempt."""
    if marker.exists():
        raise FinalEvaluationError(f"final-test marker already exists: {marker}")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({**payload, "started_at_utc": _utc_now()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def final_evaluation_is_complete(final_dir: Path) -> bool:
    """Accept completion only while the recorded final-metrics bytes remain unchanged."""
    try:
        record = json.loads((final_dir / "finalization_record.json").read_text(encoding="utf-8"))
        relative = Path(record["results"])
        if relative.is_absolute() or ".." in relative.parts or record.get("status") != "complete":
            return False
        results = (final_dir / relative).resolve()
        results.relative_to(final_dir.resolve())
        return (
            results.is_file()
            and results.stat().st_size == record["results_bytes"]
            and _sha256_file(results) == record["results_sha256"]
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args(argv)
    finalize(args.repo.resolve(), args.dataset.resolve(), args.workspace.resolve())
    return 0


def finalize(repo: Path, dataset: Path, workspace: Path) -> None:
    final_dir = workspace / "final"
    completed = final_dir / "finalization_record.json"
    if completed.is_file():
        if final_evaluation_is_complete(final_dir):
            print(f"SKIP completed final evaluation: {completed}")
            return
        raise FinalEvaluationError(f"final evidence exists but is hash-mismatched: {completed}")

    lock = InputLock(**json.loads((workspace / "inputs" / "input_lock.json").read_text()))
    git_sha, git_dirty = _git_provenance(repo)
    if git_dirty or git_sha != lock.git_sha:
        raise FinalEvaluationError("Git state no longer matches the pre-training input lock")
    protocol_spec = _load_spec(repo / "configs" / "paired_protocol.yaml")
    protocol = build_paired_protocol(
        discover_converted_samples(dataset), PairedProtocolConfig(**protocol_spec["protocol"])
    )
    _verify_frozen_hashes(protocol, protocol_spec)
    if (
        protocol.dataset_sha256 != lock.dataset_sha256
        or protocol.manifest_sha256 != lock.manifest_sha256
    ):
        raise FinalEvaluationError("dataset or manifest no longer matches the input lock")
    runtime = workspace / "runtime_data"
    render_runtime_datasets(dataset, protocol, runtime)

    run_dirs: dict[tuple[str, int], Path] = {}
    validation: dict[int, dict[str, Any]] = {}
    for arm in ARMS:
        for seed in protocol.config.training_seeds:
            run_dir = workspace / "runs" / arm / f"seed{seed}"
            if not run_is_complete(run_dir, lock):
                raise FinalEvaluationError(
                    f"run is incomplete or hash-mismatched: {arm} seed={seed}"
                )
            run_dirs[(arm, seed)] = run_dir
            if arm == "grouped":
                validation[seed] = json.loads(
                    (run_dir / "metrics" / "validation.json").read_text(encoding="utf-8")
                )

    config = yaml.safe_load((repo / "configs" / "final_evaluation.yaml").read_text())
    deployment_seed = choose_grouped_deployment_seed(validation)
    final_dir.mkdir(parents=True, exist_ok=True)
    selection = {
        "arm": "grouped",
        "seed": deployment_seed,
        "criterion": "highest grouped validation map50_95; lower seed tie-break",
        "selected_before_final_test": True,
        "validation_by_seed": {
            str(seed): metrics["map50_95"] for seed, metrics in sorted(validation.items())
        },
    }
    (final_dir / "deployment_selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    begin_final_test_once(
        final_dir / "FINAL_TEST_STARTED.json",
        {
            "git_sha": lock.git_sha,
            "dataset_sha256": lock.dataset_sha256,
            "manifest_sha256": lock.manifest_sha256,
            "final_test_stems": list(protocol.final_test),
            "deployment_selection": selection,
        },
    )

    runs: dict[str, dict[str, Any]] = {}
    for (arm, seed), run_dir in run_dirs.items():
        key = f"{arm}-seed{seed}"
        runs[key] = _evaluate_run(
            run_dir / "weights" / "best.pt", runtime / arm / "data.yaml", config
        )
    result = {
        "schema_version": "1.0",
        "status": "complete",
        "git_sha": lock.git_sha,
        "dataset_sha256": lock.dataset_sha256,
        "manifest_sha256": lock.manifest_sha256,
        "final_test_images": len(protocol.final_test),
        "final_test_board": protocol.config.final_board,
        "operating_point": {
            "confidence": config["operating_confidence"],
            "match_iou": config["match_iou"],
            "source": "configs/final_evaluation.yaml (frozen before test)",
        },
        "deployment_selection": selection,
        "runs": runs,
        "aggregate": _aggregate(runs, tuple(protocol.config.training_seeds), config),
        "limitations": [
            "The common final test contains one PCB template board and 30 images.",
            "Confidence intervals use image resampling and do not estimate between-board variance.",
            "This controlled study measures observed same-board sibling exposure "
            "sensitivity, not all leakage modes.",
        ],
        "completed_at_utc": _utc_now(),
    }
    results_path = final_dir / "final_metrics.json"
    results_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    completed.write_text(
        json.dumps(
            {
                "status": "complete",
                "completed_at_utc": result["completed_at_utc"],
                "results": "final_metrics.json",
                "results_sha256": _sha256_file(results_path),
                "results_bytes": results_path.stat().st_size,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"FINAL EVALUATION COMPLETE: {results_path}")


def _evaluate_run(weights: Path, data_yaml: Path, config: dict[str, Any]) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=config["imgsz"],
        conf=config["ap_confidence"],
        iou=config["validation_iou"],
        plots=False,
        verbose=False,
    )
    image_paths = [
        Path(line)
        for line in Path(yaml.safe_load(data_yaml.read_text())["test"]).read_text().splitlines()
    ]
    per_image: dict[str, dict[str, float | int]] = {}
    for image_path in image_paths:
        label_path = _label_path(image_path)
        with Image.open(image_path) as image:
            gt = load_yolo_labels(label_path, *image.size)
        prediction = model.predict(
            source=str(image_path),
            imgsz=config["imgsz"],
            conf=config["operating_confidence"],
            verbose=False,
        )[0]
        match = greedy_match(boxes_from_ultralytics(prediction), gt, iou_thr=config["match_iou"])
        per_image[image_path.stem] = {
            "tp": len(match.tp),
            "fp": len(match.fp),
            "fn": len(match.fn),
            "precision": match.precision,
            "recall": match.recall,
            "f1": match.f1,
        }
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": {
            model.names[index]: {
                "ap50": float(metrics.box.ap50[index]),
                "ap50_95": float(metrics.box.ap[index]),
                "precision": float(metrics.box.p[index]),
                "recall": float(metrics.box.r[index]),
            }
            for index in range(len(model.names))
        },
        "per_image": per_image,
        "fp_per_image": statistics.fmean(row["fp"] for row in per_image.values()),
        "fp_per_board": sum(row["fp"] for row in per_image.values()),
    }


def _aggregate(
    runs: dict[str, dict[str, Any]], seeds: tuple[int, ...], config: dict[str, Any]
) -> dict[str, Any]:
    by_arm: dict[str, Any] = {}
    for arm in ARMS:
        arm_runs = [runs[f"{arm}-seed{seed}"] for seed in seeds]
        by_arm[arm] = {
            metric: _mean_std([float(run[metric]) for run in arm_runs])
            for metric in ("map50", "map50_95", "fp_per_image", "fp_per_board")
        }
        by_arm[arm]["per_class"] = {
            class_name: {
                metric: _mean_std([float(run["per_class"][class_name][metric]) for run in arm_runs])
                for metric in ("ap50", "ap50_95", "precision", "recall")
            }
            for class_name in CLASSES
        }

    stems = sorted(next(iter(runs.values()))["per_image"])
    paired: dict[str, Any] = {}
    for metric in ("recall", "f1", "fp"):
        deltas = []
        for stem in stems:
            grouped = statistics.fmean(
                runs[f"grouped-seed{seed}"]["per_image"][stem][metric] for seed in seeds
            )
            leaky = statistics.fmean(
                runs[f"leaky_control-seed{seed}"]["per_image"][stem][metric] for seed in seeds
            )
            deltas.append(leaky - grouped)
        paired[f"leaky_minus_grouped_{metric}"] = paired_bootstrap_ci(
            deltas,
            n_resamples=config["bootstrap_resamples"],
            seed=config["bootstrap_seed"],
        )
    return {"by_arm": by_arm, "paired_bootstrap": paired}


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as exc:
        raise FinalEvaluationError(f"image path lacks an images/ component: {image_path}") from exc
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def _mean_std(values: list[float]) -> dict[str, float | int]:
    return {
        "n_seeds": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _quantile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
