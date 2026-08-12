from __future__ import annotations

from pathlib import Path

from app.evidence import build_app_state
from app.theme import APP_CSS
from app.ui import build_demo

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
