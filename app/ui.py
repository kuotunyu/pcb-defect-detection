# ruff: noqa: E501
"""Gradio composition for the PCB review workstation portfolio."""

from __future__ import annotations

import base64
import html
from pathlib import Path

import gradio as gr

from app.models import AppState, EvidenceSummary

REPOSITORY_URL = "https://github.com/kuotunyu/pcb-defect-detection"
APP_DIR = Path(__file__).resolve().parent


def _preview_data_uri() -> str:
    encoded = base64.b64encode(
        (APP_DIR / "assets" / "pcb-workstation-preview.svg").read_bytes()
    ).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _header_html() -> str:
    return f"""
    <div class="pcb-nav-wrap">
      <nav class="pcb-nav pcb-shell" aria-label="主要導覽">
        <a class="pcb-brand" href="#hero">
          <span class="pcb-brand-mark">PCB</span>
          <span class="pcb-brand-copy"><strong>PCB Defect Intelligence</strong><span>Evidence-driven inspection portfolio</span></span>
        </a>
        <div class="pcb-nav-links">
          <a href="#workstation">工作站</a><a href="#evidence">模型證據</a><a href="#defect-taxonomy">瑕疵類別</a>
          <a class="pcb-github" href="{REPOSITORY_URL}" target="_blank" rel="noreferrer">GitHub ↗</a>
        </div>
      </nav>
    </div>
    """


def _hero_html(state: AppState) -> str:
    integrity = html.escape(state.status_title)
    return f"""
    <section class="pcb-hero" aria-labelledby="hero-title">
      <div class="pcb-shell pcb-hero-grid">
        <div class="pcb-hero-copy">
          <div class="pcb-eyebrow"><span class="pcb-eyebrow-dot"></span>Leakage-aware Computer Vision Portfolio</div>
          <h1 id="hero-title">從資料切分到 <em>Deployment Gate</em>，完整呈現 PCB 瑕疵偵測工程。</h1>
          <p class="pcb-lead">以 YOLO26n 建構六類裸板瑕疵偵測，透過 Board-level protocol 揭露資料洩漏造成的效能膨脹，並用 committed evidence 驗證模型發布邊界。</p>
          <div class="pcb-actions"><a class="pcb-button pcb-button-primary" href="#workstation">查看複核工作站</a><a class="pcb-button" href="#evidence">瀏覽工程證據</a></div>
          <div class="pcb-integrity"><span class="pcb-integrity-icon">●</span><span><strong>{integrity}</strong> · 不下載 floating model，也不把未通過 gate 的 artifact 包裝成 public inference。</span></div>
        </div>
        <div class="pcb-preview-frame"><img src="{_preview_data_uri()}" alt="PCB 人工複核工作站介面示意，非模型輸出"></div>
      </div>
    </section>
    """


def _metric_card(label: str, value: str, context: str) -> str:
    return f"<article class='pcb-kpi'><span class='pcb-kpi-label'>{html.escape(label)}</span><strong class='pcb-kpi-value'>{html.escape(value)}</strong><span class='pcb-kpi-context'>{html.escape(context)}</span></article>"


def _kpis_html(evidence: EvidenceSummary | None) -> str:
    if evidence is None:
        cards = _metric_card("Committed evidence", "Unavailable", "請檢查 reports schema 與檔案完整性")
    else:
        cards = "".join(
            (
                _metric_card("Defect Classes", f"{evidence.defect_classes} 類", "PCB bare-board defect taxonomy"),
                _metric_card(evidence.leakage_effect.label, evidence.leakage_effect.display_value, evidence.leakage_effect.context),
                _metric_card(evidence.grouped_map50.label, evidence.grouped_map50.display_value, evidence.grouped_map50.context),
                _metric_card(evidence.ort_cuda_p50.label, evidence.ort_cuda_p50.display_value, evidence.ort_cuda_p50.context),
            )
        )
    return f"<section class='pcb-kpi-area'><div class='pcb-shell pcb-kpi-grid'>{cards}</div></section>"


def _workstation_html(state: AppState) -> str:
    status_title = html.escape(state.status_title)
    status_detail = html.escape(state.status_detail)
    return f"""
    <section class="pcb-section pcb-surface" aria-labelledby="workstation-title">
      <div class="pcb-shell">
        <div class="pcb-section-head"><div><span class="pcb-section-kicker">Review workstation</span><h2 id="workstation-title">把模型輸出放進可複核的工作流程</h2><p>原圖、標註結果、model version 與 promotion status 在同一個視線範圍；目前以 evidence mode 誠實展示產品介面。</p></div></div>
        <div class="pcb-workstation">
          <div class="pcb-workstation-visual"><img src="{_preview_data_uri()}" alt="原創 PCB 複核介面插畫，非模型輸出"></div>
          <aside class="pcb-review-panel">
            <div class="pcb-status-card"><span class="pcb-status-label">PROMOTION GATE</span><strong>BLOCKED</strong><p>{status_detail}</p></div>
            <div class="pcb-review-card"><h3>複核摘要</h3><div class="pcb-review-row"><span>Mode</span><b>{status_title}</b></div><div class="pcb-review-row"><span>Model family</span><b>YOLO26n</b></div><div class="pcb-review-row"><span>Supported classes</span><b>6 defect classes</b></div><div class="pcb-review-row"><span>Public inference</span><b>Unavailable</b></div></div>
            <p class="pcb-honesty-note"><strong>介面示意 · 非模型輸出</strong><br>上游影像授權未確認，因此 public repository 不包含 HRIPCB demo media。</p>
          </aside>
        </div>
      </div>
    </section>
    """


def _evidence_html(evidence: EvidenceSummary | None) -> str:
    grouped = evidence.grouped_map50.display_value if evidence else "Unavailable"
    leakage = evidence.leakage_effect.display_value if evidence else "Unavailable"
    parity = "PASS" if evidence and evidence.strict_parity_passed else "BLOCKED"
    return f"""
    <section class="pcb-section" aria-labelledby="evidence-title">
      <div class="pcb-shell">
        <div class="pcb-section-head"><div><span class="pcb-section-kicker">Committed evidence</span><h2 id="evidence-title">不是只展示漂亮的 bounding boxes</h2><p>每個數字都有 frozen protocol、hash-bound artifact 與 limitation；失敗的 gate 也完整保留。</p></div></div>
        <div class="pcb-evidence-grid">
          <article class="pcb-card"><span class="pcb-evidence-number">01</span><h3>Board-level Split</h3><p>以 PCB Board ID 做資料切分，避免同板 sibling image 同時落入 train 與 test。</p><div class="pcb-evidence-result"><span>Protocol frozen · PASS</span><a href="{REPOSITORY_URL}/blob/main/reports/protocol/paired_split_manifest.json" target="_blank">Evidence ↗</a></div></article>
          <article class="pcb-card"><span class="pcb-evidence-number">02</span><h3>Paired Evaluation</h3><p>Grouped 與 leaky-control 共用 final test 與三組 seeds，量化 same-board exposure effect。</p><div class="pcb-evidence-result"><span>{leakage} · mAP50 {grouped}</span><a href="{REPOSITORY_URL}/blob/main/reports/paired_a100/final_metrics.json" target="_blank">Evidence ↗</a></div></article>
          <article class="pcb-card pcb-card-blocked"><span class="pcb-evidence-number">03</span><h3>Promotion Gate</h3><p>Aggregate fidelity passed，但 strict per-box prediction parity failed，因此不發布 public model artifact。</p><div class="pcb-evidence-result"><span>Strict parity · {parity}</span><a href="{REPOSITORY_URL}/blob/main/reports/backend_parity_l4.json" target="_blank">Evidence ↗</a></div></article>
        </div>
      </div>
    </section>
    """


DEFECTS = (
    ("missing_hole", "漏孔", "預期鑽孔缺失，可能影響插件、導通或後續裝配。"),
    ("mouse_bite", "鼠咬", "板邊或導體輪廓出現不規則缺口，需要確認是否超出製程容許值。"),
    ("open_circuit", "斷路", "導電路徑中斷，可能造成訊號或電源無法通過。"),
    ("short", "短路", "不應相連的導體產生橋接，可能導致電性失效。"),
    ("spur", "毛刺", "導體邊緣出現多餘突出，可能降低線距或造成可靠度風險。"),
    ("spurious_copper", "多餘銅箔", "非設計區域殘留銅材，需要進一步人工判讀影響範圍。"),
)


def _defects_html() -> str:
    cards = "".join(
        f"<article class='pcb-defect'><span class='pcb-defect-code'>{code}</span><h3>{name}</h3><p>{description}</p></article>"
        for code, name, description in DEFECTS
    )
    return f"""
    <section class="pcb-section pcb-surface" aria-labelledby="defect-title"><div class="pcb-shell"><div class="pcb-section-head"><div><span class="pcb-section-kicker">Defect taxonomy</span><h2 id="defect-title">六種 PCB 裸板瑕疵</h2><p>介面以 canonical class order 呈現，中文名稱協助快速判讀，class identifier 保留原文。</p></div></div><div class="pcb-defect-grid">{cards}</div></div></section>
    """


def _project_links_html() -> str:
    return f"""
    <section class="pcb-section"><div class="pcb-shell"><div class="pcb-project-panel"><div><span class="pcb-section-kicker" style="color:#d8e6dc">REPRODUCIBLE RESEARCH PACKAGE</span><h2>深入檢視 protocol、model card 與發布證據</h2><p>從資料 fingerprint、paired experiment 到 L4 backend parity，所有公開宣稱都指向 committed evidence。</p></div><div class="pcb-project-links"><a href="{REPOSITORY_URL}" target="_blank">Repository ↗</a><a href="{REPOSITORY_URL}/blob/main/docs/model-card.md" target="_blank">Model Card ↗</a><a href="{REPOSITORY_URL}/blob/main/docs/research-package.md" target="_blank">Research Package ↗</a></div></div><footer class="pcb-footer"><span>PCB Defect Intelligence · AGPL-3.0-or-later</span><span>Designed for evidence-first Computer Vision review</span></footer></div></section>
    """


def build_demo(state: AppState, service: object | None = None) -> gr.Blocks:
    """Build the portfolio shell; live controls are wired only in LIVE mode later."""

    del service
    with gr.Blocks(
        title="PCB Defect Intelligence",
        fill_width=True,
        analytics_enabled=False,
    ) as demo:
        gr.HTML(_header_html(), elem_id="app-header", container=False)
        gr.HTML(_hero_html(state), elem_id="hero", container=False)
        gr.HTML(_kpis_html(state.evidence), elem_id="kpi-strip", container=False)
        gr.HTML(_workstation_html(state), elem_id="workstation", container=False)
        gr.HTML(_evidence_html(state.evidence), elem_id="evidence", container=False)
        gr.HTML(_defects_html(), elem_id="defect-taxonomy", container=False)
        gr.HTML(_project_links_html(), elem_id="project-links", container=False)
    return demo
