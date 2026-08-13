# ruff: noqa: E501
"""Gradio composition for the PCB review workstation portfolio."""

from __future__ import annotations

import base64
import html
import os
from dataclasses import replace
from pathlib import Path

import gradio as gr
from PIL import Image

from app.evidence import build_app_state
from app.inference import CLASSES, Detection, InferenceService
from app.models import AppMode, AppState, EvidenceSummary

REPOSITORY_URL = "https://github.com/kuotunyu/pcb-defect-detection"
APP_DIR = Path(__file__).resolve().parent


def _preview_data_uri() -> str:
    encoded = base64.b64encode(
        (APP_DIR / "assets" / "pcb-workstation-preview.svg").read_bytes()
    ).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _header_html() -> str:
    return f"""
    <a class="pcb-skip-link" href="#main-content">跳至主要內容</a>
    <div class="pcb-nav-wrap">
      <nav class="pcb-nav pcb-shell" aria-label="主要導覽">
        <a class="pcb-brand" href="#hero">
          <span class="pcb-brand-mark">PCB</span>
          <span class="pcb-brand-copy"><strong>PCB Defect Intelligence</strong><span>Evidence-driven inspection portfolio</span></span>
        </a>
        <div class="pcb-nav-links">
          <a href="#workstation">工作站</a><a href="#evidence">模型證據</a><a href="#defect-taxonomy">瑕疵類別</a>
          <a class="pcb-github" href="{REPOSITORY_URL}" target="_blank" rel="noopener noreferrer">GitHub ↗</a>
        </div>
      </nav>
    </div>
    """


def _hero_html(state: AppState) -> str:
    integrity = html.escape(state.status_title)
    return f"""
    <section id="main-content" class="pcb-hero" aria-labelledby="hero-title">
      <div class="pcb-shell pcb-hero-grid">
        <div class="pcb-hero-copy">
          <h1 id="hero-title"><span class="pcb-title-line">從資料切分到</span><span class="pcb-title-line">Deployment Gate</span><span class="pcb-title-line">完整呈現 PCB</span><span class="pcb-title-line">瑕疵偵測工程</span></h1>
          <p class="pcb-lead">以 YOLO26n 建構六類裸板瑕疵偵測，透過 Board-level protocol 揭露資料洩漏造成的效能膨脹，並用 committed evidence 驗證模型發布邊界。</p>
          <div class="pcb-actions"><a class="pcb-button pcb-button-primary" href="#workstation">查看複核工作站</a><a class="pcb-button" href="#evidence">瀏覽工程證據</a></div>
          <div class="pcb-integrity"><span class="pcb-integrity-icon">●</span><span><strong>{integrity}</strong> · 不下載 floating model，也不把未通過 gate 的 artifact 包裝成 public inference。</span></div>
        </div>
        <div class="pcb-preview-frame"><img src="{_preview_data_uri()}" alt="PCB 人工複核工作站介面示意，非模型輸出" width="1200" height="720" fetchpriority="high" decoding="async"></div>
      </div>
    </section>
    """


def _metric_card(label: str, value: str, context: str) -> str:
    return f"<div class='pcb-kpi'><dt class='pcb-kpi-label'>{html.escape(label)}</dt><dd><strong class='pcb-kpi-value'>{html.escape(value)}</strong><span class='pcb-kpi-context'>{html.escape(context)}</span></dd></div>"


def _kpis_html(evidence: EvidenceSummary | None) -> str:
    if evidence is None:
        cards = _metric_card("Committed evidence", "無法載入", "請檢查 reports schema 與檔案完整性")
    else:
        cards = "".join(
            (
                _metric_card(
                    "Defect Classes",
                    f"{evidence.defect_classes} 類",
                    "PCB bare-board defect taxonomy",
                ),
                _metric_card(
                    evidence.leakage_effect.label,
                    evidence.leakage_effect.display_value,
                    evidence.leakage_effect.context,
                ),
                _metric_card(
                    evidence.grouped_map50.label,
                    evidence.grouped_map50.display_value,
                    evidence.grouped_map50.context,
                ),
                _metric_card(
                    evidence.ort_cuda_p50.label,
                    evidence.ort_cuda_p50.display_value,
                    evidence.ort_cuda_p50.context,
                ),
            )
        )
    return f"<section class='pcb-kpi-area' aria-label='核心工程指標'><dl class='pcb-shell pcb-kpi-grid'>{cards}</dl></section>"


def _workstation_html(state: AppState) -> str:
    status_title = html.escape(state.status_title)
    status_detail = html.escape(state.status_detail)
    if state.mode is AppMode.DEGRADED:
        gate_status = "UNAVAILABLE"
        gate_explanation = "Committed evidence 或 model runtime 無法完成驗證；inference 已停用。"
        technical_detail = (
            f'<p class="pcb-contract-detail"><span>Technical detail</span>{status_detail}</p>'
        )
    else:
        gate_status = "BLOCKED"
        gate_explanation = "Aggregate fidelity 已通過；strict L4 backend prediction-parity gate 未通過，因此不發布 public ONNX artifact、hosted model revision 或 inference endpoint。"
        technical_detail = ""
    return f"""
    <section class="pcb-section pcb-surface" aria-labelledby="workstation-title">
      <div class="pcb-shell">
        <div class="pcb-section-head"><div><h2 id="workstation-title">把模型輸出放進可複核的工作流程</h2><p>原圖、標註結果、model version 與 promotion status 在同一個視線範圍；目前以 evidence mode 誠實展示產品介面。</p></div></div>
        <div class="pcb-workstation">
          <div class="pcb-workstation-visual"><img src="{_preview_data_uri()}" alt="原創 PCB 複核介面插畫，非模型輸出" width="1200" height="720" loading="lazy" decoding="async"></div>
          <aside class="pcb-review-panel">
            <div class="pcb-status-card"><span class="pcb-status-label">PROMOTION GATE</span><strong>{gate_status}</strong><p>{gate_explanation}</p>{technical_detail}</div>
            <div class="pcb-review-card"><h3>複核摘要</h3><div class="pcb-review-row"><span>Mode</span><b>{status_title}</b></div><div class="pcb-review-row"><span>Model family</span><b>YOLO26n</b></div><div class="pcb-review-row"><span>Supported classes</span><b>6 defect classes</b></div><div class="pcb-review-row"><span>Public inference</span><b>Unavailable（未發布）</b></div></div>
            <p class="pcb-honesty-note"><strong>介面示意 · 非模型輸出</strong><br>上游影像授權未確認，因此 public repository 不包含 HRIPCB demo media。</p>
          </aside>
        </div>
      </div>
    </section>
    """


def _evidence_html(evidence: EvidenceSummary | None, state: AppState) -> str:
    grouped = evidence.grouped_map50.display_value if evidence else "Unavailable"
    leakage = evidence.leakage_effect.display_value if evidence else "Unavailable"
    if state.mode is AppMode.LIVE:
        parity = "PASS"
        gate_copy = "Release-approved strict per-box prediction parity evidence is committed."
        gate_class = "pcb-card"
    elif evidence is None:
        parity = "UNVERIFIED"
        gate_copy = (
            "Committed evidence 無法完整載入，因此無法驗證 Promotion Gate；inference 已停用。"
        )
        gate_class = "pcb-card pcb-card-blocked"
    else:
        parity = "BLOCKED"
        gate_copy = "Aggregate fidelity passed，但 strict per-box prediction parity failed，因此不發布 public model artifact。"
        gate_class = "pcb-card pcb-card-blocked"
    return f"""
    <section class="pcb-section" aria-labelledby="evidence-title">
      <div class="pcb-shell">
        <div class="pcb-section-head"><div><h2 id="evidence-title">不是只展示漂亮的 bounding boxes</h2><p>每個數字都有 frozen protocol、hash-bound artifact 與 limitation；失敗的 gate 也完整保留。</p></div></div>
        <div class="pcb-evidence-grid">
          <article class="pcb-card"><span class="pcb-evidence-number">01</span><h3>Board-level Split</h3><p>以 PCB Board ID 做資料切分，避免同板 sibling image 同時落入 train 與 test。</p><div class="pcb-evidence-result"><span>Protocol frozen · PASS</span><a href="{REPOSITORY_URL}/blob/main/reports/protocol/paired_split_manifest.json" target="_blank" rel="noopener noreferrer">Evidence ↗</a></div></article>
          <article class="pcb-card"><span class="pcb-evidence-number">02</span><h3>Paired Evaluation</h3><p>Grouped 與 leaky-control 共用 final test 與三組 seeds，量化 same-board exposure effect。</p><div class="pcb-evidence-result"><span>{leakage} · mAP50 {grouped}</span><a href="{REPOSITORY_URL}/blob/main/reports/paired_a100/final_metrics.json" target="_blank" rel="noopener noreferrer">Evidence ↗</a></div></article>
          <article class="{gate_class}"><span class="pcb-evidence-number">03</span><h3>Promotion Gate</h3><p>{gate_copy}</p><div class="pcb-evidence-result"><span>Strict parity · {parity}</span><a href="{REPOSITORY_URL}/blob/main/reports/backend_parity_l4.json" target="_blank" rel="noopener noreferrer">Evidence ↗</a></div></article>
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
    <section class="pcb-section pcb-surface" aria-labelledby="defect-title"><div class="pcb-shell"><div class="pcb-section-head"><div><h2 id="defect-title">六種 PCB 裸板瑕疵</h2><p>介面以 canonical class order 呈現，中文名稱協助快速判讀，class identifier 保留原文。</p></div></div><div class="pcb-defect-grid">{cards}</div></div></section>
    """


def _project_links_html() -> str:
    return f"""
    <section class="pcb-section"><div class="pcb-shell"><div class="pcb-project-panel"><div><h2>深入檢視 protocol、model card 與發布證據</h2><p>從資料 fingerprint、paired experiment 到 L4 backend parity，所有公開宣稱都指向 committed evidence。</p></div><div class="pcb-project-links"><a href="{REPOSITORY_URL}" target="_blank" rel="noopener noreferrer">Repository ↗</a><a href="{REPOSITORY_URL}/blob/main/docs/model-card.md" target="_blank" rel="noopener noreferrer">Model Card ↗</a><a href="{REPOSITORY_URL}/blob/main/docs/research-package.md" target="_blank" rel="noopener noreferrer">Research Package ↗</a></div></div><footer class="pcb-footer"><span>PCB Defect Intelligence · AGPL-3.0-or-later</span><span>Designed for evidence-first Computer Vision review</span></footer></div></section>
    """


def _annotations(detections: tuple[Detection, ...]) -> list[tuple[tuple[int, ...], str]]:
    return [
        (
            tuple(round(value) for value in detection.xyxy),
            f"{CLASSES[detection.cls_id]} · {detection.confidence:.2f}",
        )
        for detection in detections
    ]


def _detection_rows(detections: tuple[Detection, ...]) -> list[list[object]]:
    return [
        [
            CLASSES[detection.cls_id],
            round(detection.confidence, 3),
            *[round(value, 1) for value in detection.xyxy],
        ]
        for detection in detections
    ]


def run_inference_ui(
    image: Image.Image | None,
    confidence: float,
    service: InferenceService,
) -> tuple[tuple[Image.Image, list[tuple[tuple[int, ...], str]]] | None, str, list[list[object]]]:
    """Adapt the inference service result to stable, zh-TW Gradio outputs."""

    if image is None:
        return None, "請先選擇 PCB 圖片，再執行偵測。", []
    try:
        result = service.run(image, confidence)
    except Exception as error:  # keep the selected input and surface the runtime failure
        return (image, []), f"偵測失敗：{html.escape(str(error))}", []

    if not result.detections:
        summary = (
            "未偵測到高於目前 confidence threshold 的瑕疵；"
            "此結果不代表 PCB 無缺陷。"
            f"\n\nEnd-to-end latency：{result.latency_ms:.0f} ms"
        )
    else:
        summary = (
            f"偵測到 **{len(result.detections)}** 個候選瑕疵，請進行人工複核。"
            f"\n\nEnd-to-end latency：{result.latency_ms:.0f} ms"
        )
    return (image, _annotations(result.detections)), summary, _detection_rows(result.detections)


def _render_live_workstation(state: AppState, service: InferenceService) -> None:
    repo_id = html.escape(str(state.contract.get("hf_repo_id", "release-approved artifact")))
    revision = html.escape(str(state.contract.get("hf_revision", "unavailable"))[:12])
    sha256 = html.escape(str(state.contract.get("onnx_sha256", "unavailable"))[:12])
    runtime = html.escape(service.runtime_label)
    gr.HTML(
        f"""
        <section class="pcb-section pcb-surface" aria-labelledby="live-workstation-title">
          <div class="pcb-shell"><div class="pcb-section-head"><div>
            <h2 id="live-workstation-title">上傳 PCB 影像並複核模型候選瑕疵</h2>
            <p>模型 artifact、revision 與 SHA-256 已在啟動時通過 committed contract 驗證。</p>
          </div></div><div class="pcb-live-provenance"><span><b>PROMOTED</b> Promotion Gate</span><span><b>{repo_id}</b> Model</span><span><b>{revision}</b> Revision</span><span><b>{sha256}</b> ONNX SHA-256</span><span><b>{runtime}</b> Runtime</span></div></div>
        </section>
        """,
        elem_id="workstation",
        container=False,
    )
    with gr.Row(elem_id="live-workstation", elem_classes="pcb-shell live-workstation-grid"):
        with gr.Column(elem_classes="live-control-panel"):
            image_input = gr.Image(type="pil", label="PCB image", elem_id="pcb-image-input")
            confidence = gr.Slider(
                0.05,
                0.90,
                value=0.25,
                step=0.01,
                label="Confidence threshold",
                elem_id="confidence-threshold",
            )
            run_button = gr.Button("執行偵測", variant="primary", elem_id="run-inference")
        with gr.Column(elem_classes="live-result-panel"):
            annotated = gr.AnnotatedImage(label="偵測結果", elem_id="detection-result")
            summary = gr.Markdown(
                "選擇影像後執行偵測；所有模型輸出仍需人工複核。",
                elem_id="inference-summary",
            )
            table = gr.Dataframe(
                headers=["class", "confidence", "x1", "y1", "x2", "y2"],
                datatype=["str", "number", "number", "number", "number", "number"],
                interactive=False,
                label="Detection details",
                elem_id="detection-table",
            )
    run_button.click(
        fn=lambda selected_image, threshold: run_inference_ui(
            selected_image,
            threshold,
            service,
        ),
        inputs=[image_input, confidence],
        outputs=[annotated, summary, table],
    )


def build_demo(state: AppState, service: InferenceService | None = None) -> gr.Blocks:
    """Build the complete portfolio and expose inference only in verified LIVE mode."""

    with gr.Blocks(
        title="PCB Defect Intelligence",
        fill_width=True,
        analytics_enabled=False,
    ) as demo:
        gr.HTML(_header_html(), elem_id="app-header", container=False)
        gr.HTML(_hero_html(state), elem_id="hero", container=False)
        gr.HTML(_kpis_html(state.evidence), elem_id="kpi-strip", container=False)
        if state.mode is AppMode.LIVE and service is not None:
            _render_live_workstation(state, service)
        else:
            gr.HTML(_workstation_html(state), elem_id="workstation", container=False)
        gr.HTML(_evidence_html(state.evidence, state), elem_id="evidence", container=False)
        gr.HTML(_defects_html(), elem_id="defect-taxonomy", container=False)
        gr.HTML(_project_links_html(), elem_id="project-links", container=False)
    return demo


def create_demo(repo_root: Path | None = None) -> gr.Blocks:
    """Build startup state, hash-verify an optional model, and compose the UI."""

    root = repo_root or Path(__file__).resolve().parents[1]
    state = build_app_state(root)
    service = None
    if state.mode is AppMode.LIVE:
        try:
            service = InferenceService.from_contract(
                state.contract,
                os.environ.get("MODEL_PATH_OVERRIDE"),
            )
        except Exception as error:
            message = str(error)
            state = replace(
                state,
                mode=AppMode.DEGRADED,
                inference_enabled=False,
                status_title="Model unavailable",
                status_detail=message,
                errors=state.errors + (message,),
            )
    return build_demo(state, service)
