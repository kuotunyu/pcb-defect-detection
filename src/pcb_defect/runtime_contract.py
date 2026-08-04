"""Hermetic Ultralytics and ONNX Runtime environment contract."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from importlib import metadata
from typing import Any

EXPECTED_ORT_VERSION = "1.26.0"
ORT_DISTRIBUTIONS = ("onnxruntime", "onnxruntime-gpu")


class RuntimeContractError(RuntimeError):
    """The installed ONNX runtime cannot produce trustworthy evidence."""


def configure_hermetic_ultralytics() -> None:
    """Force Ultralytics to leave the locked environment unchanged."""
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"] = "1"


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def onnxruntime_state(require_cuda_provider: bool = False) -> dict[str, object]:
    """Validate and return a JSON-serializable snapshot of the exact ORT state."""
    expected_by_platform = {"linux": "onnxruntime-gpu", "win32": "onnxruntime"}
    expected_distribution = expected_by_platform.get(sys.platform)
    if expected_distribution is None:
        raise RuntimeContractError(f"unsupported ONNX Runtime platform: {sys.platform}")

    versions = {name: _distribution_version(name) for name in ORT_DISTRIBUTIONS}
    observed = versions[expected_distribution]
    if observed != EXPECTED_ORT_VERSION:
        raise RuntimeContractError(
            f"{expected_distribution} must equal {EXPECTED_ORT_VERSION}, found {observed!r}"
        )
    conflicting = next(name for name in ORT_DISTRIBUTIONS if name != expected_distribution)
    if versions[conflicting] is not None:
        raise RuntimeContractError(
            "conflicting ONNX Runtime distribution is installed: "
            f"{conflicting}=={versions[conflicting]}"
        )

    ort: Any = importlib.import_module("onnxruntime")
    module_version = getattr(ort, "__version__", None)
    if module_version != EXPECTED_ORT_VERSION:
        raise RuntimeContractError(
            f"onnxruntime module must equal {EXPECTED_ORT_VERSION}, found {module_version!r}"
        )
    providers = list(ort.get_available_providers())
    if "CPUExecutionProvider" not in providers:
        raise RuntimeContractError("ONNX Runtime is missing CPUExecutionProvider")
    if require_cuda_provider and "CUDAExecutionProvider" not in providers:
        raise RuntimeContractError("ONNX Runtime is missing CUDAExecutionProvider")

    return {
        "platform": sys.platform,
        "expected_distribution": expected_distribution,
        "distribution_versions": versions,
        "module_version": module_version,
        "available_providers": providers,
        "cuda_required": require_cuda_provider,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda-provider", action="store_true")
    args = parser.parse_args(argv)
    configure_hermetic_ultralytics()
    print(json.dumps(onnxruntime_state(args.require_cuda_provider), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
