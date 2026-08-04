"""Hash-pinned ONNX demo; blocked until the deployment contract passes."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
CONTRACT_PATH = APP_DIR / "model_contract.json"
MODEL_PATH_OVERRIDE = os.environ.get("MODEL_PATH_OVERRIDE")

CLASSES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
IMG_SIZE = 640
PAD_VALUE = 114
DEFAULT_CONF = 0.25


@dataclass(frozen=True)
class LetterboxInfo:
    gain: float
    pad_left: float
    pad_top: float


@dataclass(frozen=True)
class Detection:
    cls_id: int
    xyxy: tuple[float, float, float, float]
    conf: float


def _load_contract() -> dict:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("schema_version") != "1.0":
        raise RuntimeError("Unsupported model contract schema")
    return contract


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_model(contract: dict) -> Path:
    if contract.get("status") != "passed":
        raise RuntimeError(f"Model promotion is blocked: {contract.get('reason', 'no reason')}")
    expected = contract.get("onnx_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError("Passed model contract lacks a valid ONNX SHA-256")
    if MODEL_PATH_OVERRIDE:
        path = Path(MODEL_PATH_OVERRIDE).resolve()
    else:
        repo_id = contract.get("hf_repo_id")
        revision = contract.get("hf_revision")
        if not repo_id or not revision:
            raise RuntimeError("Passed model contract must pin a repository and immutable revision")
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=contract["filename"],
                revision=revision,
            )
        )
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
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3
    )
    return canvas, LetterboxInfo(gain, left, top)


def preprocess(image: Image.Image) -> tuple[np.ndarray, LetterboxInfo]:
    canvas, info = letterbox(image)
    chw = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(np.expand_dims(chw, axis=0)), info


def postprocess(
    output: np.ndarray, info: LetterboxInfo, orig_size: tuple[int, int], conf: float
) -> list[Detection]:
    rows = output[0]
    rows = rows[rows[:, 4] >= conf]
    orig_width, orig_height = orig_size
    detections = []
    for x1, y1, x2, y2, score, cls in rows:
        ox1 = float(max(0.0, min((x1 - info.pad_left) / info.gain, orig_width)))
        oy1 = float(max(0.0, min((y1 - info.pad_top) / info.gain, orig_height)))
        ox2 = float(max(0.0, min((x2 - info.pad_left) / info.gain, orig_width)))
        oy2 = float(max(0.0, min((y2 - info.pad_top) / info.gain, orig_height)))
        detections.append(Detection(int(cls), (ox1, oy1, ox2, oy2), float(score)))
    return detections


def _annotations(detections: list[Detection]) -> list[tuple[tuple[int, int, int, int], str]]:
    return [
        (
            tuple(round(value) for value in detection.xyxy),
            f"{CLASSES[detection.cls_id]} {detection.conf:.2f}",
        )
        for detection in detections
    ]


def _table(detections: list[Detection]) -> list[list]:
    return [
        [
            CLASSES[detection.cls_id],
            round(detection.conf, 3),
            *[round(value, 1) for value in detection.xyxy],
        ]
        for detection in detections
    ]


CONTRACT = _load_contract()
SESSION = None
INPUT_NAME = None
STARTUP_ERROR = None
if CONTRACT.get("status") == "passed":
    try:
        import onnxruntime as ort

        model_path = _resolve_model(CONTRACT)
        SESSION = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        INPUT_NAME = SESSION.get_inputs()[0].name
    except Exception as error:  # displayed as a deployment failure, never silently bypassed
        STARTUP_ERROR = str(error)


def run_inference(image: Image.Image | None, conf: float):
    if image is None:
        return None, "", []
    if SESSION is None or INPUT_NAME is None:
        raise gr.Error(STARTUP_ERROR or CONTRACT.get("reason", "Model promotion is blocked"))
    started = time.perf_counter()
    batch, info = preprocess(image)
    (raw_output,) = SESSION.run(None, {INPUT_NAME: batch})
    elapsed_ms = (time.perf_counter() - started) * 1000
    detections = postprocess(raw_output, info, image.size, conf)
    return (
        (image, _annotations(detections)),
        f"End-to-end latency: {elapsed_ms:.0f} ms",
        _table(detections),
    )


with gr.Blocks(title="Leakage-aware PCB defect detection") as demo:
    gr.Markdown("# Leakage-aware PCB defect detection")
    if CONTRACT.get("status") != "passed" or STARTUP_ERROR:
        reason = STARTUP_ERROR or CONTRACT.get("reason", "No gate-passed model is available")
        gr.Markdown(
            "## Deployment blocked\n\n"
            f"{reason}\n\n"
            "This is intentional: the app will not download a floating model or serve an "
            "artifact that lacks a passing hash-pinned fidelity/parity contract."
        )
    else:
        gr.Markdown(
            "Upload a PCB image. The model file and Hub revision are pinned by the committed "
            "deployment contract and verified by SHA-256 at startup."
        )
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="PCB image")
                confidence = gr.Slider(0.05, 0.90, value=DEFAULT_CONF, step=0.01)
                run_button = gr.Button("Run inference", variant="primary")
            with gr.Column():
                annotated = gr.AnnotatedImage(label="Detections")
                latency = gr.Markdown()
                table = gr.Dataframe(
                    headers=["class", "confidence", "x1", "y1", "x2", "y2"],
                    interactive=False,
                )
        run_button.click(
            run_inference,
            inputs=[image_input, confidence],
            outputs=[annotated, latency, table],
        )


if __name__ == "__main__":
    demo.launch()
