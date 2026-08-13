# README 圖解敘事設計

日期：2026-08-13

狀態：已核准設計，待實作計畫

## 目標

讓首次造訪 GitHub 的招募者或工程師在 30 秒內看懂專案價值，並能在 3 分鐘內確認這不是只有漂亮 bounding boxes 的展示，而是一套具備資料防洩漏、可追溯證據、部署門控與 fail-closed UI 的 Computer Vision 工程作品。

README 仍以正體中文為主，Computer Vision、MLOps 與 deployment 專有名詞保留原文。圖解必須增加資訊價值，不重複既有段落，也不能暗示目前存在 hosted inference、public model artifact 或 production SLA。

## 現況與設計缺口

README 已有：

- 完整的 PCB 人工複核工作站 screenshot 與 Recorded evidence mode 說明。
- 30 秒證據索引、committed reports、研究限制與可重現命令。
- 兩張細節完整的 Mermaid flowchart，分別描述板級成對實驗與 ONNX／NVIDIA L4 部署管線。

目前缺少：

- 一張能快速說明使用者、public repository、private evidence production 與 external archive 關係的 System Context Diagram。
- 一張能把 Data Contract、Experiment、Deployment Validation、Evidence Presentation 串成同一架構的 Architecture Diagram。
- 一張準確呈現 app 在 evidence 缺失、promotion blocked、contract mismatch、artifact 驗證失敗與 live runtime 成功時如何 fail closed 的 Sequence Diagram。

## 選定方案：證據敘事型

採用三張主圖建立閱讀主線，保留既有兩張技術 Pipeline 作為可展開的 Deep Dive：

1. System Context & Trust Boundary：回答「誰使用它、公開什麼、什麼留在私有環境」。
2. Evidence-first Architecture：回答「資料與證據如何穿過四個工程層級」。
3. Fail-closed Workstation Startup：回答「為什麼 gate 未通過仍可安全展示 UI，以及何時才可能進入 live inference」。
4. Deep Dive：保留既有 Board-level paired experiment 與 ONNX／L4 flowchart，但降低它們在主閱讀路徑中的垂直占用。

不採用五張圖全部攤開的「完整技術展示型」，因為會拉長 README 並重複 Pipeline 資訊；也不採用只留兩張圖的「極簡招募者型」，因為會弱化本專案最有辨識度的研究與證據深度。

## README 閱讀順序

1. Badges、UI screenshot 與作品定位。
2. 短版 project summary，先說明 leakage effect、deployment evidence 與限制。
3. `系統全貌與證據邊界`：System Context Diagram。
4. `30 秒證據索引`：讓讀者可立即核對 committed artifacts。
5. `Evidence-first 系統架構`：Architecture Diagram 與精簡的四項關鍵特性。
6. `Fail-closed 工作站啟動時序`：Sequence Diagram。
7. `Deep Dive：實驗與部署 Pipeline`：既有兩張詳細 flowchart，使用 `<details>` 降低預設篇幅。
8. 後續資料分割、評測數字、瑕疵類別、快速開始、專案結構、引用與授權維持現有邏輯。

## Diagram 1：System Context & Trust Boundary

使用 GitHub 原生 Mermaid `flowchart LR`，不用 C4 專用語法，以降低 GitHub Mermaid 版本相容風險。

### 節點與邊界

- Actor：`GitHub 訪客／技術面試官`。
- Public Portfolio boundary：
  - `README 與 UI screenshots`
  - `PCB 人工複核工作站`
  - `Committed evidence`（reports、model card、limitations）
  - `GitHub Release + Zenodo DOI`
- Private Evidence Production boundary：
  - `HRIPCB licensed media`
  - `A100 paired training`
  - `NVIDIA L4 backend benchmark`
  - `Private weights / ONNX / TensorRT engine`

### 關係

- 訪客閱讀 README、開啟 workstation，並從 UI／README 追到 committed evidence。
- private environment 只 promotion 經整理、去識別化且 hash-bound 的 metadata／reports 到 public repository。
- public repository 發布 source、evidence 與版本化 release 到 GitHub／Zenodo；不發布 dataset media、weights、ONNX 或 TensorRT engine。
- 圖中明確標示 `No hosted inference` 與 `No public model artifact`，避免把作品集 UI 誤讀為公開推論服務。

## Diagram 2：Evidence-first Architecture

使用 Mermaid `flowchart LR` 與四個 subgraph，讓每層只有一個清楚職責：

1. Data Contract
   - `paired_protocol.yaml`
   - board-level split manifest
   - dataset／manifest SHA-256
2. Experiment & Evaluation
   - Grouped vs Leaky Control
   - A100 × Seeds 42／43／44
   - common Board 08 final test
3. Deployment Validation
   - PyTorch → ONNX
   - aggregate fidelity
   - NVIDIA L4 latency
   - strict per-box parity gate
4. Evidence Presentation
   - committed JSON／Markdown reports
   - `app.evidence.build_app_state`
   - Recorded evidence／Degraded／Live UI mode

主要資料流由左至右。strict parity failure 仍可輸出 failed-gate evidence，但只能進入 Recorded evidence mode；只有 passed contract、committed strict parity 與 runtime artifact verification 一致時，才可能進入 Live mode。

## Diagram 3：Fail-closed Workstation Startup

使用 Mermaid `sequenceDiagram`，participants 為：

- Visitor
- Gradio UI
- App State Builder
- Evidence Loader
- Model Contract
- ONNX Runtime

### 時序與分支

1. 建立 UI 時先載入 committed reports 與 `app/model_contract.json`，不先下載模型。
2. committed evidence 缺失或 schema 不符：回傳 `DEGRADED`，隱藏 inference controls。
3. contract 不是 `passed`：回傳 `EVIDENCE`，顯示 Recorded evidence mode，且不建立 ONNX session。
4. contract 宣告 passed，但 committed strict parity evidence 為 failed：視為 release evidence mismatch，回傳 `DEGRADED`。
5. 狀態可進入 Live candidate 時，才解析 immutable revision／override、驗證 ONNX SHA-256 並建立 ONNX Runtime session。
6. artifact、hash、下載或 session 建立失敗：降級為 `DEGRADED`，不顯示 inference controls。
7. 全部成功：建立 `LIVE` workstation；使用者上傳影像後才執行 deterministic preprocessing、inference 與 postprocessing。

這張圖必須反映 `app/evidence.py`、`app/inference.py` 與 `app/ui.py` 的實際條件，不把理想化未實作流程畫進 README。

## Deep Dive 編排

既有兩張 Mermaid flowchart 的實驗數字、資料邊界與 failed gate 敘事皆保留。每張圖各放入語意清楚的 `<details>`：

- `展開：PCB 板級防洩漏與成對實驗流程`
- `展開：ONNX fidelity 與 NVIDIA L4 多後端部署管線`

summary 下先用一句話說明讀者會看到什麼，再放 Mermaid。若 GitHub 對 `<details>` 內 Mermaid 的實際 rendering 不穩定，實作時改為保留展開顯示，不以自訂 JavaScript 或外部圖片繞過。

## 視覺與文字規則

- 所有 diagram 使用相同語意色彩：
  - 深松綠：public／verified evidence
  - 沙金色：processing／validation
  - 柔粉棕：blocked／failed gate
  - 淺灰綠：context／supporting artifacts
- Mermaid 預設字級設定為 17–18 px，避免 GitHub 顯示時過小。
- 節點文字以正體中文為主，必要的 class、file、runtime 與 protocol 名稱保留原文。
- 節點不放完整段落；每個節點最多約三行，長說明放在 diagram 前後的 prose。
- 不使用 emoji 或依賴特殊 icon font；箭頭方向與文字標籤同時傳達意義，不只靠顏色。
- 每張 Mermaid 加上 `accTitle` 與 `accDescr`，讓 assistive technology 能理解圖意。
- 顏色需在 GitHub light／dark theme 皆維持足夠對比，文字色不得只依賴 theme default。

## 內容與誠信邊界

圖解只能重述目前 committed evidence 與程式實作：

- Grouped mAP50 `63.30%`、leakage effect `+21.3 pp`、ORT CUDA p50 `20.28 ms`。
- aggregate fidelity 通過，但 strict per-box prediction parity failed。
- benchmark 是 NVIDIA L4、single-session、calibration-only，不是 production SLA。
- public repository 沒有 dataset media、public checkpoint、ONNX artifact、TensorRT engine 或 hosted inference。
- Recorded evidence mode 是目前實際狀態；Live mode 是程式已定義但需通過 contract 與 runtime verification 的受控能力。

任何圖解都不得把「程式支援的受控 Live path」描述成「目前已公開上線」。

## 驗證標準

實作完成後至少執行：

1. 檢查所有 Mermaid fenced blocks 有成對 fence，且 `flowchart`／`sequenceDiagram` 語法可被 Mermaid CLI 或等效 renderer 解析。
2. 在 GitHub-compatible Markdown preview 檢查三張主圖與 `<details>` 內既有圖。
3. 桌面寬度與窄視窗各檢查一次：節點文字可讀、沒有異常截斷、Sequence Diagram 不需理解極小字。
4. 核對每個 file path、report link、metrics 與 mode 名稱仍與 repository 相符。
5. 執行 README／documentation 相關測試與完整 CI 可行的本機驗證，確認沒有破壞既有 project contract。
6. 檢查 `git diff`，確保變更聚焦於 README 圖解與必要的敘事調整，沒有改動模型、數據或實驗宣稱。

## 不在本次範圍

- 不重做已定案的 Gradio UI。
- 不新增 hosted inference、public model upload 或 deployment endpoint。
- 不修改實驗數據、gate threshold、training recipe 或 committed reports。
- 不以外部 diagram SaaS、不可追蹤圖片或需要額外瀏覽器 script 的格式取代 Mermaid。
- 不為了圖解進行與 README 無關的程式重構。
