# README 掃讀性精簡設計

日期：2026-08-13

## 問題

README 已能完整呈現 UI、系統架構與 deployment evidence，但部分內容仍以長段落一次攤開。GitHub 訪客必須先讀完大量限定條件，才抓得到核心成果；Mermaid 目前的 22 px 字級也比正文略突出。

## 決策

採用「先結論、後證據」的 layered disclosure：首屏先給三個可掃讀的核心結論；流程改用短表格；原始 benchmark 數值與 strict parity 細節放入可展開區塊。這不是刪除限制，而是把限制放到最容易核對的位置。

### 內容層級

- 首屏專案總覽縮成三個 outcome：防洩漏實驗、部署速度、發布邊界。
- Evidence-first 四階段說明由長編號段落改為兩欄表格，每列只回答「做什麼」與「證據是什麼」。
- A100 與 L4 的主畫面保留結果表及一句判讀；raw values、bootstrap 邊界與 per-box mismatch 移入 `<details>`。
- 快速開始的 Notebook 說明改為短條列，避免跨多行 prose。

### 圖解字級

- Mermaid `themeVariables.fontSize` 由 22 px 降至 21 px。
- 不改圖解結構、節點文案與色彩，避免再次引入橫向過寬或文字過小問題。

## 必須保留的證據邊界

- `+21.3 pp`、`63.30%`、`20.28 ms` 與 `60/60` 等 committed metrics。
- 單一 held-out Board 08、image-level bootstrap 不估計 between-board uncertainty。
- aggregate fidelity 通過，但 PyTorch-reference strict per-box parity failed。
- calibration-only、no hosted inference、no public model artifact、no production SLA。

## 驗證

- README contract 先以 failing test 固定 21 px 與 layered-disclosure 結構。
- Mermaid CLI 必須成功解析並渲染全部圖解。
- 完整 pytest、Ruff 必須通過。
- 推送後在 GitHub 100% 縮放檢查首屏、架構與 benchmark 區塊。

## 非目標

- 不變更模型、評測數字、promotion gate 或 repository 授權邊界。
- 不把 README 拆成需要來回跳轉的多份文件。
- 不以隱藏細節掩飾 failed gate。
