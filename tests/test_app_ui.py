from __future__ import annotations

from pathlib import Path

import numpy as np
from app.evidence import build_app_state, load_evidence
from app.inference import InferenceService
from app.models import AppMode, AppState
from app.theme import APP_CSS
from app.ui import build_demo, run_inference_ui
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _config_text() -> str:
    return str(build_demo(build_app_state(ROOT)).get_config_file())


def test_theme_keeps_primary_copy_readable_and_honors_reduced_motion() -> None:
    css = APP_CSS.lower()

    assert "--pcb-pine: #3f5d4d" in css
    assert "font-size: 17px" in css
    assert "prefers-reduced-motion" in css


def test_evidence_mode_renders_the_complete_portfolio_without_live_action() -> None:
    text = _config_text()

    assert "PCB Defect Intelligence" in text
    assert "從資料切分到 <em>Deployment Gate</em>" in text
    assert "介面示意 · 非模型輸出" in text
    assert "執行偵測" not in text


def test_portfolio_sections_have_stable_browser_targets() -> None:
    text = _config_text()

    for element_id in (
        "hero",
        "kpi-strip",
        "workstation",
        "evidence",
        "defect-taxonomy",
        "project-links",
    ):
        assert element_id in text


def test_preview_asset_is_original_ui_artwork_without_claimed_confidence() -> None:
    svg = (ROOT / "app" / "assets" / "pcb-workstation-preview.svg").read_text(
        encoding="utf-8"
    )

    assert "介面示意 · 非模型輸出" in svg
    assert "0.9" not in svg


def test_degraded_mode_keeps_portfolio_and_shows_inline_error() -> None:
    state = AppState(
        mode=AppMode.DEGRADED,
        evidence=None,
        contract={},
        status_title="Evidence unavailable",
        status_detail="Cannot read committed evidence",
        inference_enabled=False,
        errors=("Cannot read committed evidence",),
    )

    text = str(build_demo(state).get_config_file())

    assert "PCB Defect Intelligence" in text
    assert "Evidence unavailable" in text
    assert "Cannot read committed evidence" in text
    assert "執行偵測" not in text


def test_live_mode_exposes_upload_and_inference_controls() -> None:
    state = AppState(
        mode=AppMode.LIVE,
        evidence=load_evidence(ROOT),
        contract={"schema_version": "1.0", "status": "passed"},
        status_title="Live inference",
        status_detail="Runtime verified",
        inference_enabled=True,
    )

    text = str(build_demo(state, InferenceService(_EmptySession(), "images")).get_config_file())

    assert "PCB image" in text
    assert "Confidence threshold" in text
    assert "執行偵測" in text


def test_no_detection_message_does_not_claim_the_board_is_defect_free() -> None:
    image = Image.new("RGB", (64, 64), "white")

    annotated, summary, rows = run_inference_ui(
        image,
        0.25,
        InferenceService(_EmptySession(), "images"),
    )

    assert annotated == (image, [])
    assert "未偵測到高於目前 confidence threshold 的瑕疵" in summary
    assert "不代表 PCB 無缺陷" in summary
    assert rows == []


class _EmptySession:
    def run(self, output_names, inputs):
        del output_names, inputs
        return [np.empty((1, 0, 6), dtype=np.float32)]
