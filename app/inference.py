"""Hash-pinned ONNX inference service for the optional live workstation mode."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CLASSES = ("missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper")
IMG_SIZE = 640
PAD_VALUE = 114


@dataclass(frozen=True)
class LetterboxInfo:
    gain: float
    pad_left: float
    pad_top: float


@dataclass(frozen=True)
class Detection:
    cls_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class InferenceResult:
    image: Image.Image
    detections: tuple[Detection, ...]
    latency_ms: float


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model(contract: dict, model_path_override: str | None = None) -> Path:
    """Resolve only a passed, immutable, hash-matching ONNX artifact."""

    if contract.get("schema_version") != "1.0":
        raise RuntimeError("Unsupported model contract schema")
    if contract.get("status") != "passed":
        reason = contract.get("reason", "no reason")
        raise RuntimeError(f"Model promotion is blocked: {reason}")

    expected = contract.get("onnx_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError("Passed model contract lacks a valid ONNX SHA-256")

    if model_path_override:
        path = Path(model_path_override).resolve()
    else:
        repo_id = contract.get("hf_repo_id")
        revision = contract.get("hf_revision")
        filename = contract.get("filename")
        if not repo_id or not revision or not filename:
            raise RuntimeError("Passed model contract must pin a repository and immutable revision")
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision))

    observed = _sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"ONNX SHA-256 mismatch: expected {expected}, found {observed}")
    return path


def letterbox(image: Image.Image, size: int = IMG_SIZE) -> tuple[np.ndarray, LetterboxInfo]:
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    gain = min(size / width, size / height)
    new_width, new_height = round(width * gain), round(height * gain)
    resized = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - new_width) / 2, (size - new_height) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    canvas = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(PAD_VALUE,) * 3,
    )
    return canvas, LetterboxInfo(gain, left, top)


def preprocess(image: Image.Image) -> tuple[np.ndarray, LetterboxInfo]:
    canvas, info = letterbox(image)
    chw = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(np.expand_dims(chw, axis=0)), info


def postprocess(
    output: np.ndarray,
    info: LetterboxInfo,
    original_size: tuple[int, int],
    confidence: float,
) -> list[Detection]:
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[2] != 6:
        raise ValueError(f"Unexpected model output shape: {output.shape}")
    rows = output[0]
    rows = rows[rows[:, 4] >= confidence]
    original_width, original_height = original_size
    detections: list[Detection] = []
    for x1, y1, x2, y2, score, cls_id in rows:
        box = np.asarray((x1, y1, x2, y2), dtype=np.float64)
        if not np.isfinite(box).all() or x2 < x1 or y2 < y1:
            raise ValueError("Model output contains an invalid detection box")
        if not np.isfinite(score):
            raise ValueError("Model output contains a non-finite confidence score")
        if not np.isfinite(cls_id) or cls_id != int(cls_id) or not 0 <= int(cls_id) < len(CLASSES):
            raise ValueError(f"Model output contains invalid class id: {cls_id}")
        ox1 = float(max(0.0, min((x1 - info.pad_left) / info.gain, original_width)))
        oy1 = float(max(0.0, min((y1 - info.pad_top) / info.gain, original_height)))
        ox2 = float(max(0.0, min((x2 - info.pad_left) / info.gain, original_width)))
        oy2 = float(max(0.0, min((y2 - info.pad_top) / info.gain, original_height)))
        detections.append(Detection(int(cls_id), (ox1, oy1, ox2, oy2), float(score)))
    return detections


class InferenceService:
    """Small ONNX session wrapper with deterministic preprocessing and timing."""

    def __init__(self, session: object, input_name: str):
        self._session = session
        self._input_name = input_name

    @classmethod
    def from_contract(
        cls,
        contract: dict,
        model_path_override: str | None = None,
    ) -> InferenceService:
        model_path = resolve_model(contract, model_path_override)
        import onnxruntime as ort

        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        return cls(session, session.get_inputs()[0].name)

    def run(self, image: Image.Image, confidence: float) -> InferenceResult:
        started = time.perf_counter()
        batch, info = preprocess(image)
        raw_outputs = self._session.run(None, {self._input_name: batch})
        (raw_output,) = raw_outputs
        detections = tuple(postprocess(raw_output, info, image.size, confidence))
        elapsed_ms = (time.perf_counter() - started) * 1000
        return InferenceResult(image=image, detections=detections, latency_ms=elapsed_ms)

    @property
    def runtime_label(self) -> str:
        get_providers = getattr(self._session, "get_providers", None)
        if not callable(get_providers):
            return "ONNX Runtime"
        providers = get_providers()
        provider = providers[0] if providers else "provider unavailable"
        return f"ONNX Runtime · {provider}"
