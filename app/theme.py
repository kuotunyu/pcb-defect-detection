# ruff: noqa: E501
"""Visual tokens and responsive styling for the PCB portfolio."""

from __future__ import annotations

import gradio as gr

APP_THEME = gr.themes.Base(
    primary_hue="green",
    secondary_hue="orange",
    neutral_hue="gray",
    text_size="lg",
    spacing_size="lg",
    radius_size="sm",
    font=("IBM Plex Sans", "Noto Sans TC", "Microsoft JhengHei", "system-ui", "sans-serif"),
    font_mono=("IBM Plex Mono", "Cascadia Code", "Consolas", "monospace"),
)

APP_HEAD = '<meta name="theme-color" content="#f4f3ef">'

APP_JS = r"""() => {
  document.documentElement.lang = "zh-TW";
  const liveRegion = document.getElementById("inference-summary");
  if (liveRegion) {
    liveRegion.setAttribute("aria-live", "polite");
    liveRegion.setAttribute("aria-atomic", "true");
  }
}"""


APP_CSS = """
:root {
  --pcb-pine: #3f5d4d;
  --pcb-pine-deep: #263b31;
  --pcb-sage: #738e7d;
  --pcb-mist: #e8ece7;
  --pcb-ivory: #f4f3ef;
  --pcb-paper: #fffefb;
  --pcb-peach: #e8c5b8;
  --pcb-brick: #9d6257;
  --pcb-amber: #b08a52;
  --pcb-ink: #26332e;
  --pcb-muted: #63706a;
  --pcb-line: #d5dcd6;
  --pcb-shadow: 0 18px 48px rgba(44, 61, 52, 0.11);
}

html { scroll-behavior: smooth; background-color: var(--pcb-ivory) !important; text-rendering: optimizeLegibility; }
[id] { scroll-margin-top: 92px; }
body {
  margin: 0 !important;
  overflow-x: hidden;
  color-scheme: light;
  background-color: var(--pcb-ivory) !important;
  background:
    radial-gradient(circle at 8% 0%, rgba(115, 142, 125, 0.13), transparent 30rem),
    linear-gradient(180deg, #f8f7f3 0%, var(--pcb-ivory) 100%) !important;
}
.gradio-container {
  max-width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  color: var(--pcb-ink) !important;
  font-family: "IBM Plex Sans", "Noto Sans TC", "Microsoft JhengHei", system-ui, sans-serif !important;
  font-size: 18px !important;
  line-height: 1.65 !important;
  min-height: 100vh;
  background-color: var(--pcb-ivory) !important;
  background-image:
    radial-gradient(circle at 8% 0%, rgba(115, 142, 125, 0.13), transparent 30rem),
    linear-gradient(180deg, #f8f7f3 0%, var(--pcb-ivory) 100%) !important;
}
.gradio-container > .main { padding: 0 !important; }
.gradio-container .block { margin: 0 !important; }
.gradio-container main.contain > .column { gap: 0 !important; }
#app-header > .html-container,
#hero > .html-container,
#kpi-strip > .html-container,
#workstation > .html-container,
#evidence > .html-container,
#defect-taxonomy > .html-container,
#project-links > .html-container { padding: 0 !important; }
footer, .built-with { display: none !important; }
a, button { touch-action: manipulation; -webkit-tap-highlight-color: rgba(63, 93, 77, 0.14); }
.pcb-skip-link {
  position: fixed;
  top: 10px;
  left: 10px;
  z-index: 100;
  padding: 10px 14px;
  border-radius: 4px;
  color: #fff;
  background: var(--pcb-pine-deep);
  transform: translateY(-150%);
}
.pcb-skip-link:focus { transform: translateY(0); }

.pcb-shell { width: min(1360px, calc(100% - 40px)); margin: 0 auto; }
.pcb-nav-wrap {
  position: sticky;
  top: 0;
  z-index: 40;
  border-bottom: 1px solid rgba(213, 220, 214, 0.86);
  background: rgba(255, 254, 251, 0.92);
  backdrop-filter: blur(18px);
}
.pcb-nav {
  min-height: 68px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 28px;
}
.pcb-brand { display: flex; align-items: center; gap: 12px; color: var(--pcb-ink); text-decoration: none; }
.pcb-brand-mark {
  width: 38px;
  height: 38px;
  display: grid;
  place-items: center;
  border-radius: 6px;
  color: #fff;
  background: var(--pcb-pine);
  box-shadow: 0 8px 20px rgba(63, 93, 77, 0.25);
  font: 700 14px/1 "IBM Plex Mono", monospace;
  letter-spacing: 0.04em;
}
.pcb-brand-copy strong { display: block; color: var(--pcb-ink) !important; font-size: 18px; line-height: 1.2; letter-spacing: -0.02em; }
.pcb-brand-copy span { display: block; color: var(--pcb-muted); font-size: 15px; line-height: 1.2; margin-top: 2px; }
.pcb-nav-links { display: flex; align-items: center; gap: 24px; }
.pcb-nav-links a {
  color: var(--pcb-muted);
  font-size: 17px;
  font-weight: 650;
  text-decoration: none;
  transition: color 180ms ease;
}
.pcb-nav-links a:hover, .pcb-nav-links a:focus-visible { color: var(--pcb-pine); }
.pcb-nav-links .pcb-github {
  padding: 9px 13px;
  border: 1px solid var(--pcb-line);
  border-radius: 4px;
  color: var(--pcb-ink);
  background: var(--pcb-paper);
}

.pcb-section { padding: 24px 0; }
.pcb-hero {
  position: relative;
  overflow: hidden;
  padding: 16px 0 8px;
}
.pcb-hero::before {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  opacity: 0.36;
  background-image:
    linear-gradient(rgba(63, 93, 77, 0.06) 1px, transparent 1px),
    linear-gradient(90deg, rgba(63, 93, 77, 0.06) 1px, transparent 1px);
  background-size: 34px 34px;
  mask-image: linear-gradient(to bottom, black, transparent 82%);
}
.pcb-hero-grid {
  position: relative;
  display: grid;
  grid-template-columns: minmax(0, 0.88fr) minmax(540px, 1.12fr);
  align-items: center;
  gap: 28px;
}
.pcb-hero h1 {
  max-width: 720px;
  margin: 0 0 14px;
  color: var(--pcb-ink);
  font-size: clamp(36px, 3vw, 46px);
  line-height: 1.08;
  letter-spacing: -0.04em;
  text-wrap: balance;
}
.pcb-hero h1 em { color: var(--pcb-pine); font-style: normal; }
.pcb-title-line { display: block; color: var(--pcb-ink) !important; }
.pcb-title-line:nth-child(2) { color: var(--pcb-pine) !important; }
.pcb-lead { max-width: 680px; margin: 0; color: #596760; font-size: 20px; line-height: 1.58; }
.pcb-hero-support { display: grid; grid-template-columns: max-content minmax(0, 1fr); align-items: center; gap: 16px; margin-top: 18px; }
.pcb-actions { display: flex; flex-wrap: wrap; gap: 8px; margin: 0; }
.pcb-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 46px;
  padding: 0 17px;
  border: 1px solid var(--pcb-line);
  border-radius: 4px;
  color: var(--pcb-ink);
  background: var(--pcb-paper);
  font-size: 16px;
  font-weight: 750;
  text-decoration: none;
  box-shadow: 0 6px 16px rgba(44, 61, 52, 0.06);
  transition: border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease;
}
.pcb-button:hover { transform: translateY(-2px); border-color: var(--pcb-sage); box-shadow: 0 10px 24px rgba(44, 61, 52, 0.11); }
.pcb-button-primary { color: #fff; border-color: var(--pcb-pine); background: var(--pcb-pine); }
.pcb-integrity { display: flex; align-items: flex-start; gap: 8px; margin: 0; color: var(--pcb-muted); font-size: 16px; line-height: 1.48; }
.pcb-integrity span { color: var(--pcb-muted) !important; }
.pcb-integrity strong { color: #76564e !important; }
.pcb-integrity-icon { color: var(--pcb-brick); font-size: 16px; line-height: 1.5; }

.pcb-preview-frame {
  position: relative;
  padding: 8px;
  border: 1px solid rgba(255, 255, 255, 0.11);
  border-radius: 8px;
  background: #2d3b35;
  box-shadow: 0 28px 72px rgba(38, 51, 46, 0.27);
}
.pcb-preview-frame::after {
  content: "";
  position: absolute;
  inset: auto 10% -22px;
  height: 28px;
  border-radius: 50%;
  background: rgba(46, 65, 56, 0.18);
  filter: blur(18px);
}
.pcb-preview-frame img { position: relative; z-index: 1; display: block; width: 100%; height: auto; aspect-ratio: 5 / 3; border-radius: 3px; }

.pcb-kpi-area { padding: 0 0 12px; }
.pcb-kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  margin-block: 0;
  overflow: hidden;
  border-block: 1px solid var(--pcb-line);
  background: rgba(255, 254, 251, 0.9);
}
.pcb-kpi {
  min-width: 0;
  min-height: 96px;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-content: start;
  gap: 6px 12px;
  padding: 12px 16px;
  background: transparent;
}
.pcb-kpi + .pcb-kpi { border-left: 1px solid var(--pcb-line); }
.pcb-kpi dd { display: contents; }
.pcb-kpi-label { min-width: 0; color: var(--pcb-muted); font-size: 16px; font-weight: 700; line-height: 1.4; letter-spacing: 0.01em; }
.pcb-kpi-value { grid-column: 2; grid-row: 1; color: var(--pcb-ink); font: 700 29px/1.05 "IBM Plex Mono", monospace; letter-spacing: -0.04em; white-space: nowrap; }
.pcb-kpi-context { grid-column: 1 / -1; color: var(--pcb-muted); font-size: 16px; line-height: 1.42; overflow-wrap: anywhere; }

.pcb-surface { background: rgba(255, 254, 251, 0.72); border-block: 1px solid rgba(213, 220, 214, 0.8); }
.pcb-section-head { display: flex; justify-content: space-between; align-items: end; gap: 24px; margin-bottom: 12px; }
.pcb-section h2 { margin: 0 0 6px; color: var(--pcb-ink); font-size: clamp(28px, 2.6vw, 34px); line-height: 1.2; letter-spacing: -0.032em; text-wrap: balance; }
.pcb-section-head p { max-width: 800px; margin: 0; color: var(--pcb-muted); font-size: 19px; }

.pcb-workstation {
  display: grid;
  grid-template-columns: minmax(0, 1.45fr) minmax(320px, 0.55fr);
  gap: 8px;
  padding: 8px;
  border-radius: 8px;
  background: var(--pcb-pine-deep);
  box-shadow: var(--pcb-shadow);
}
.pcb-workstation-visual { padding: 8px; border-radius: 2px; background: #e9eee9; }
.pcb-workstation-visual img { display: block; width: 100%; height: auto; aspect-ratio: 5 / 3; border-radius: 0; }
.pcb-review-panel { display: flex; flex-direction: column; gap: 8px; }
.pcb-status-card, .pcb-review-card { padding: 16px; border-radius: 0; background: #fffefb; }
.pcb-status-card { background: #59443f; color: #f8f0ed; }
.pcb-status-label { color: #e6cbc2; font: 700 15px/1.2 "IBM Plex Mono", monospace; letter-spacing: 0.05em; }
.pcb-status-card strong { display: block; margin: 7px 0 8px; font-size: 22px; }
.pcb-status-card p { margin: 0; color: #f0dfda; font-size: 17px; line-height: 1.5; overflow-wrap: anywhere; }
.pcb-review-card { flex: 1; }
.pcb-review-card h3 { color: var(--pcb-ink) !important; margin: 0 0 10px; font-size: 20px; }
.pcb-review-row { display: flex; justify-content: space-between; gap: 18px; padding: 8px 0; border-top: 1px solid #e7e9e6; color: var(--pcb-muted); font-size: 17px; }
.pcb-review-row span { color: var(--pcb-muted) !important; }
.pcb-review-row b { color: var(--pcb-ink); text-align: right; }
.pcb-contract-detail { margin-top: 12px !important; padding-top: 11px; border-top: 1px solid rgba(240, 223, 218, 0.24); font-size: 15px !important; overflow-wrap: anywhere; }
.pcb-contract-detail span { display: block; margin-bottom: 4px; color: #e6cbc2 !important; font-weight: 700; }
.pcb-honesty-note { margin: 0; padding: 12px 14px; border-left: 1px solid var(--pcb-sage); color: #50665a; background: #edf3ee; font-size: 17px; }
.pcb-live-provenance {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 9px;
  margin-top: 18px;
}
.pcb-live-provenance span {
  padding: 11px 12px;
  border: 1px solid var(--pcb-line);
  border-radius: 0;
  color: var(--pcb-muted) !important;
  background: var(--pcb-paper);
  font-size: 16px;
}
.pcb-live-provenance b { display: block; overflow-wrap: anywhere; color: var(--pcb-ink) !important; }

#live-workstation {
  width: min(1360px, calc(100% - 40px));
  margin: -26px auto 48px !important;
  padding: 16px !important;
  gap: 16px !important;
  border-radius: 8px;
  background: var(--pcb-pine-deep);
  box-shadow: var(--pcb-shadow);
}
#live-workstation .live-control-panel,
#live-workstation .live-result-panel {
  gap: 13px !important;
  padding: 18px !important;
  border-radius: 2px;
  background: var(--pcb-paper);
}
#live-workstation .form,
#live-workstation .block {
  border-color: var(--pcb-line) !important;
  box-shadow: none !important;
}
#run-inference button {
  min-height: 48px;
  border: 0 !important;
  color: #fff !important;
  background: var(--pcb-pine) !important;
  font-size: 16px !important;
  font-weight: 750 !important;
}

.pcb-evidence-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  overflow: hidden;
  border-block: 1px solid var(--pcb-line);
  background: var(--pcb-paper);
}
.pcb-card {
  padding: 18px;
  background: transparent;
}
.pcb-card + .pcb-card { border-left: 1px solid var(--pcb-line); }
.pcb-evidence-number { color: #4d6b59; font: 700 16px/1 "IBM Plex Mono", monospace; }
.pcb-card h3 { margin: 12px 0 7px; color: var(--pcb-ink); font-size: 20px; line-height: 1.3; }
.pcb-card p { margin: 0; color: var(--pcb-muted); font-size: 18px; line-height: 1.55; }
.pcb-evidence-result { display: flex; align-items: center; justify-content: space-between; gap: 14px; margin-top: 14px; padding-top: 11px; border-top: 1px solid #e6e9e6; color: var(--pcb-pine); font-size: 16px; font-weight: 750; }
.pcb-card-blocked .pcb-evidence-number { color: #82584f; }
.pcb-card-blocked .pcb-evidence-result { color: var(--pcb-brick); }
.pcb-evidence-result span, .pcb-evidence-result a { color: inherit !important; }

.pcb-defect-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); overflow: hidden; border-block: 1px solid var(--pcb-line); background: var(--pcb-paper); }
.pcb-defect { min-width: 0; min-height: 132px; padding: 17px 18px; border-right: 1px solid var(--pcb-line); border-bottom: 1px solid var(--pcb-line); background: transparent; }
.pcb-defect:nth-child(3n) { border-right: 0; }
.pcb-defect:nth-last-child(-n + 3) { border-bottom: 0; }
.pcb-defect-code { color: var(--pcb-pine); font: 700 15px/1.2 "IBM Plex Mono", monospace; }
.pcb-defect h3 { margin: 9px 0 5px; font-size: 20px; }
.pcb-defect p { margin: 0; color: var(--pcb-muted); font-size: 18px; line-height: 1.5; }

.pcb-project-panel {
  display: grid;
  grid-template-columns: 1fr auto;
  align-items: center;
  gap: 28px;
  padding: 22px;
  border-radius: 6px;
  color: #f6f7f5;
  background: linear-gradient(125deg, #2d4338, #496553);
  box-shadow: var(--pcb-shadow);
}
.pcb-project-panel h2 { margin: 0 0 8px; color: #fff; }
.pcb-project-panel p { max-width: 780px; margin: 0; color: #dce6df; font-size: 18px; }
.pcb-project-links { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 9px; }
.pcb-project-links a { padding: 10px 12px; border: 1px solid rgba(255, 255, 255, 0.25); border-radius: 3px; color: #fff; font-size: 16px; font-weight: 700; text-decoration: none; background: rgba(255, 255, 255, 0.07); transition: background-color 180ms ease, border-color 180ms ease; }
.pcb-project-links a:hover, .pcb-project-links a:focus-visible { border-color: rgba(255, 255, 255, 0.55); background: rgba(255, 255, 255, 0.14); }
.pcb-footer { display: flex; justify-content: space-between; gap: 24px; padding: 14px 0 16px; color: var(--pcb-muted); font-size: 16px; }

@keyframes pcb-rise {
  from { opacity: 0.62; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
.pcb-hero-copy { animation: pcb-rise 520ms cubic-bezier(0.16, 1, 0.3, 1) both; }
.pcb-preview-frame { animation: pcb-rise 620ms 70ms cubic-bezier(0.16, 1, 0.3, 1) both; }

a:focus-visible, button:focus-visible, input:focus-visible { outline: 3px solid rgba(115, 142, 125, 0.46) !important; outline-offset: 3px !important; }

@media (max-width: 1099px) {
  .pcb-shell { width: min(100% - 32px, 980px); }
  .pcb-hero-grid { grid-template-columns: 1fr; }
  .pcb-hero-copy { max-width: 820px; }
  .pcb-preview-frame { max-width: 840px; }
  .pcb-kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .pcb-kpi + .pcb-kpi { border-left: 1px solid var(--pcb-line); }
  .pcb-kpi:nth-child(odd) { border-left: 0; }
  .pcb-kpi:nth-child(n + 3) { border-top: 1px solid var(--pcb-line); }
  .pcb-workstation { grid-template-columns: 1fr; }
  .pcb-live-provenance { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .pcb-review-panel { display: grid; grid-template-columns: 0.8fr 1.2fr; }
  .pcb-honesty-note { grid-column: 1 / -1; }
  .pcb-evidence-grid { grid-template-columns: 1fr; }
  .pcb-card + .pcb-card { border-top: 1px solid var(--pcb-line); border-left: 0; }
  .pcb-defect-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .pcb-defect, .pcb-defect:nth-child(3n), .pcb-defect:nth-last-child(-n + 3) { border-right: 1px solid var(--pcb-line); border-bottom: 1px solid var(--pcb-line); }
  .pcb-defect:nth-child(2n) { border-right: 0; }
  .pcb-defect:nth-last-child(-n + 2) { border-bottom: 0; }
}

@media (max-width: 767px) {
  .pcb-shell { width: calc(100% - 24px); }
  .pcb-nav { min-height: 64px; }
  .pcb-brand-copy span, .pcb-nav-links a:not(.pcb-github) { display: none; }
  .pcb-nav-links { gap: 0; }
  .pcb-hero { padding: 16px 0 12px; }
  .pcb-hero h1 { font-size: clamp(32px, 7.8vw, 36px); }
  .pcb-lead { font-size: 19px; }
  .pcb-section { padding: 20px 0; }
  .pcb-kpi-area { padding-bottom: 12px; }
  .pcb-live-provenance { grid-template-columns: 1fr; }
  .pcb-kpi { min-height: 0; }
  .pcb-section-head { align-items: flex-start; flex-direction: column; }
  .pcb-workstation { padding: 9px; }
  .pcb-project-panel { grid-template-columns: 1fr; padding: 24px 20px; }
  .pcb-project-links { justify-content: flex-start; }
  .pcb-footer { flex-direction: column; gap: 7px; }
  #live-workstation { width: calc(100% - 24px); margin-top: -20px !important; padding: 9px !important; }
}

@media (max-width: 519px) {
  .pcb-hero-support { grid-template-columns: 1fr; gap: 12px; }
  .pcb-kpi-grid, .pcb-review-panel, .pcb-defect-grid { grid-template-columns: 1fr; }
  .pcb-kpi + .pcb-kpi, .pcb-kpi:nth-child(odd) { border-left: 0; }
  .pcb-kpi:nth-child(n + 2) { border-top: 1px solid var(--pcb-line); }
  .pcb-defect, .pcb-defect:nth-child(2n), .pcb-defect:nth-child(3n), .pcb-defect:nth-last-child(-n + 2), .pcb-defect:nth-last-child(-n + 3) { border-right: 0; border-bottom: 1px solid var(--pcb-line); }
  .pcb-defect:last-child { border-bottom: 0; }
}

@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
"""
