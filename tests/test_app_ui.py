from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import app.theme as theme
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
    assert "font-size: 18px !important" in css
    assert "font-size: clamp(36px, 3vw, 46px)" in css
    assert ".pcb-lead { max-width: 680px;" in css
    assert "font-size: 20px" in css
    assert "prefers-reduced-motion" in css


def test_tablet_layout_uses_available_width_before_switching_to_single_column() -> None:
    css = APP_CSS.lower()

    assert ".pcb-shell { width: calc(100% - 24px); }" in css
    assert "width: min(100% - 24px, 620px)" not in css
    assert "@media (max-width: 519px)" in css


def test_hero_support_and_kpis_use_horizontal_space_before_adding_height() -> None:
    text = _config_text()
    css = APP_CSS.lower()

    assert '<div class="pcb-hero-support">' in text
    assert 'class="pcb-integrity-copy"' in text
    assert ".pcb-kpi dd { display: contents; }" in css
    assert "grid-column: 2" in css
    kpi_value_rule = next(line for line in css.splitlines() if line.startswith(".pcb-kpi-value {"))
    assert 'font-family: "ibm plex sans"' in kpi_value_rule
    assert "font-variant-numeric: tabular-nums" in kpi_value_rule


def test_navigation_and_hero_actions_use_comfortable_control_type() -> None:
    css = APP_CSS.lower()
    nav_rule = css.split(".pcb-nav-links a {", 1)[1].split("}", 1)[0]
    button_rule = css.split(".pcb-button {", 1)[1].split("}", 1)[0]

    assert "font-size: 18px" in nav_rule
    assert "font-size: 17px" in button_rule


def test_theme_forces_the_approved_light_canvas_in_dark_system_mode() -> None:
    css = APP_CSS.lower()

    assert "background: transparent !important" not in css
    assert "background-color: var(--pcb-ivory) !important" in css
    assert ".pcb-integrity span { color: var(--pcb-muted) !important; }" in css
    assert ".pcb-title-line { display: block; color: var(--pcb-ink) !important; }" in css
    assert ".pcb-review-card h3 { color: var(--pcb-ink) !important;" in css
    assert "scroll-margin-top" in css


def test_evidence_mode_renders_the_complete_portfolio_without_live_action() -> None:
    text = _config_text()

    assert "PCB Defect Intelligence" in text
    assert "Deployment Gate" in text
    assert "介面示意 · 非模型輸出" in text
    assert "Aggregate fidelity 已通過" in text
    assert "this metadata-only portfolio release candidate intentionally" not in text
    assert "執行偵測" not in text


def test_document_metadata_prefers_zh_tw_and_announces_async_results() -> None:
    head = getattr(theme, "APP_HEAD", "")
    javascript = getattr(theme, "APP_JS", "")

    assert 'name="theme-color" content="#f4f3ef"' in head
    assert 'document.documentElement.lang = "zh-TW"' in javascript
    assert 'setAttribute("aria-live", "polite")' in javascript


def test_preview_images_reserve_space_and_use_intentional_loading_priority() -> None:
    text = _config_text()

    assert text.count('width="1200" height="720"') == 2
    assert 'fetchpriority="high"' in text
    assert 'loading="lazy"' in text


def test_hero_title_uses_punctuation_free_editorial_line_breaks() -> None:
    text = _config_text()

    for line in ("從資料切分到", "Deployment Gate", "完整呈現 PCB", "瑕疵偵測工程"):
        assert f'<span class="pcb-title-line">{line}</span>' in text
    assert "Deployment Gate，" not in text
    assert "瑕疵偵測工程。" not in text


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
    assert "跳至主要內容" in text


def test_preview_asset_is_original_ui_artwork_without_claimed_confidence() -> None:
    svg = (ROOT / "app" / "assets" / "pcb-workstation-preview.svg").read_text(encoding="utf-8")

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
    assert "Aggregate fidelity passed" not in text
    assert "無法驗證 Promotion Gate" in text
    assert "執行偵測" not in text


def test_live_mode_exposes_upload_and_inference_controls() -> None:
    state = AppState(
        mode=AppMode.LIVE,
        evidence=load_evidence(ROOT),
        contract={
            "schema_version": "1.0",
            "status": "passed",
            "onnx_sha256": "a" * 64,
            "hf_repo_id": "owner/model",
            "hf_revision": "b" * 40,
        },
        status_title="Live inference",
        status_detail="Runtime verified",
        inference_enabled=True,
    )

    text = str(build_demo(state, InferenceService(_EmptySession(), "images")).get_config_file())

    assert "PCB image" in text
    assert "Confidence threshold" in text
    assert "執行偵測" in text
    assert "PROMOTED" in text
    assert "owner/model" in text
    assert "bbbbbbbbbbbb" in text
    assert "aaaaaaaaaaaa" in text
    assert "ONNX Runtime" in text
    assert "Strict parity · PASS" in text


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

    def get_providers(self):
        return ["CPUExecutionProvider"]


def test_malformed_output_becomes_inline_error_and_preserves_input() -> None:
    image = Image.new("RGB", (64, 64), "white")
    service = InferenceService(_MalformedClassSession(), "images")

    annotated, summary, rows = run_inference_ui(image, 0.25, service)

    assert annotated == (image, [])
    assert "偵測失敗" in summary
    assert "class id" in summary
    assert rows == []


class _MalformedClassSession(_EmptySession):
    def run(self, output_names, inputs):
        del output_names, inputs
        return [np.array([[[1, 1, 10, 10, 0.8, 9]]], dtype=np.float32)]


def test_declared_direct_app_entry_point_builds_successfully() -> None:
    result = subprocess.run(
        [sys.executable, "app/app.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "PCB Defect Intelligence" in result.stdout
