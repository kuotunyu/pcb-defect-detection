from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from app.inference import (
    InferenceService,
    LetterboxInfo,
    postprocess,
    preprocess,
    resolve_model,
)
from PIL import Image


def test_preprocess_returns_nchw_float32() -> None:
    batch, info = preprocess(Image.new("RGB", (320, 160), "white"))

    assert batch.shape == (1, 3, 640, 640)
    assert batch.dtype == np.float32
    assert info.gain == 2.0
    assert info.pad_top == 160


def test_postprocess_maps_boxes_to_original_coordinates() -> None:
    output = np.array([[[20, 180, 220, 380, 0.9, 2]]], dtype=np.float32)

    result = postprocess(output, LetterboxInfo(2.0, 0, 160), (320, 160), 0.25)

    assert result[0].cls_id == 2
    assert result[0].xyxy == pytest.approx((10, 10, 110, 110))


def test_blocked_contract_cannot_create_service() -> None:
    with pytest.raises(RuntimeError, match="promotion is blocked"):
        InferenceService.from_contract(
            {"schema_version": "1.0", "status": "blocked", "reason": "parity failed"}
        )


def test_model_override_requires_matching_sha256(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"verified-model")
    contract = {
        "schema_version": "1.0",
        "status": "passed",
        "onnx_sha256": hashlib.sha256(b"different-model").hexdigest(),
    }

    with pytest.raises(RuntimeError, match="ONNX SHA-256 mismatch"):
        resolve_model(contract, str(model))
