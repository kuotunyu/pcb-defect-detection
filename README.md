# pcb-defect-detection

[![CI](https://github.com/kuotunyu/pcb-defect-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/kuotunyu/pcb-defect-detection/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21877496.svg)](https://doi.org/10.5281/zenodo.21877496)
![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?logo=pytorch&logoColor=white)
![Ultralytics YOLO26n](https://img.shields.io/badge/YOLO26n-Object%20Detection-blue?logo=ultralytics&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-passing-success)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-lightgrey.svg)](LICENSE)

## PCB 人工複核工作站

![PCB 人工複核工作站](docs/assets/ui-workstation-desktop.png)

新版 UI 將模型、評估與發布證據整理成一個可閱讀的 **PCB 人工複核工作站**：第一個畫面同時呈現產品用途、關鍵 metrics、model version 與 Promotion Gate。介面目前以 **Recorded evidence** 模式呈現 committed metrics 與介面示意，不宣稱提供 hosted inference。

- **影像複核視角**：原圖／標註檢視並排，保留 confidence、latency 與複核摘要的產品結構。
- **Evidence-first**：`+21.3 pp` leakage effect、`63.30%` grouped mAP50 與 `20.28 ms` L4 calibration latency 皆直接來自 committed reports。
- **Fail closed**：aggregate fidelity 通過但 strict per-box parity failed；UI 保留完整作品集內容，但不解鎖未通過 contract 的 inference。

### 本機啟動

```powershell
uv run --locked --no-editable --extra app python -m app.app
```

> **Windows 路徑提醒**：Python 3.11 在 CP950 locale 下讀取 editable-install `.pth` 時，可能無法解析含中文字的 checkout path。建議將 repository clone 到純 ASCII 路徑，並沿用上方 `--no-editable` 啟動方式；CI 也使用 non-editable install。

目前啟動後會進入 Recorded evidence mode。未來只有在 `model_contract.json` 宣告 `passed`、artifact SHA-256 相符且 ONNX session 建立成功時，才會顯示上傳與 `執行偵測` 控制項。

<details>
<summary>Mobile layout</summary>

![PCB 人工複核工作站 mobile](docs/assets/ui-workstation-mobile.png)

</details>

---

本專案針對印刷電路板 (PCB) 瑕疵檢測場景，建立基於 **YOLO26n** 之嚴格板級資料防洩漏 (Board-level Leakage) 評測基準與多後端部署驗證管線：在 frozen paired protocol 下，針對單一 held-out Board 08 的 30 張 final-test images，觀察到 same-board sibling exposure 對應 `21.3` 個百分點的 mAP50 差距。此結果限於固定 dataset 與 training recipe，不估計 between-board 或 production generalization。專案另提供 aggregate ONNX fidelity、strict per-box prediction parity，以及 NVIDIA L4 上 PyTorch、ONNX Runtime CUDA 與 TensorRT 的 calibration-only 部署證據；其中 aggregate fidelity 通過，但 strict per-box parity gate 未通過。

## 系統全貌與證據邊界

公開 repository 提供可閱讀、可核對的作品集與 committed evidence；受授權資料、模型 artifacts 與 GPU 執行環境維持在 private evidence-production boundary。兩者只透過 hash-bound reports 與版本化 metadata 連接。

```mermaid
%%{init: {'themeVariables': {'fontSize': '24px'}}}%%
flowchart TB
    accTitle: PCB 瑕疵偵測系統全貌與公開證據邊界
    accDescr: GitHub 訪客由公開作品集閱讀 committed evidence；受授權資料、GPU 工作與模型 artifacts 留在 private boundary，只發布 hash-bound reports。

    Visitor(["GitHub 訪客／技術面試官"])
    Portfolio["Public Portfolio — README · 複核工作站"]
    Evidence["Committed Evidence — reports · hashes"]
    Private["Private Evidence Production — private data · GPU · artifacts"]
    Archive["GitHub Release + Zenodo DOI — versioned archive"]

    Visitor -->|瀏覽與核對| Portfolio
    Portfolio --> Evidence --> Archive
    Private -.->|hash-bound reports| Evidence
    Private -.->|No hosted inference · No public model artifact| Portfolio

    classDef actor fill:#F3F0E8,stroke:#35594A,stroke-width:2px,color:#26352F
    classDef public fill:#35594A,stroke:#26352F,stroke-width:2px,color:#FFFFFF
    classDef evidence fill:#DCE7DF,stroke:#35594A,stroke-width:2px,color:#26352F
    classDef private fill:#EDE2C8,stroke:#9A7438,stroke-width:2px,color:#26352F
    classDef blocked fill:#F3E2DD,stroke:#785650,stroke-width:2px,color:#3F2E2B

    class Visitor actor
    class Portfolio public
    class Evidence,Archive evidence
    class Private private
```

## 30 秒證據索引

| 招募者要核對的內容 | Committed evidence |
|---|---|
| Same-board exposure 的 paired effect（3 seeds、共同 final test） | [`reports/paired_a100/final_metrics.json`](reports/paired_a100/final_metrics.json) |
| Frozen split、board policy 與 dataset fingerprint | [`reports/protocol/paired_split_manifest.json`](reports/protocol/paired_split_manifest.json) |
| Hash-pinned ONNX fidelity 與 standalone parity gate | [`reports/paired_a100/deployment_gate.public.json`](reports/paired_a100/deployment_gate.public.json) |
| NVIDIA L4 latency、raw timings、fidelity 與 strict parity | [`reports/benchmark_l4.json`](reports/benchmark_l4.json) · [`reports/benchmark_l4_raw.json`](reports/benchmark_l4_raw.json) · [`reports/backend_parity_l4.json`](reports/backend_parity_l4.json) · [`reports/benchmark_l4.md`](reports/benchmark_l4.md) |

---

## Evidence-first 系統架構

系統不是把 training、benchmark 與 UI 當成彼此無關的展示，而是讓每一層輸出下一層可驗證的 contract。failed gate 仍會成為 evidence，但不會被轉譯成已發布模型。

```mermaid
%%{init: {'themeVariables': {'fontSize': '24px'}}}%%
flowchart TB
    accTitle: PCB 瑕疵偵測的 evidence-first 四層架構
    accDescr: Frozen data contract 依序驅動 paired evaluation、deployment gate 與 evidence presentation；失敗的 gate 只進入 Recorded evidence 或 Degraded mode。

    Data["01 · Data Contract — frozen split · SHA-256"]
    Evaluation["02 · Paired Evaluation — Grouped vs Leaky"]
    Gate["03 · Deployment Gate — fidelity · strict parity"]
    Presentation["04 · Evidence Presentation — Recorded · Degraded · Live"]

    Data -->|固定資料邊界| Evaluation
    Evaluation -->|committed metrics| Gate
    Gate -->|狀態與限制| Presentation

    classDef contract fill:#DCE7DF,stroke:#35594A,stroke-width:2px,color:#26352F
    classDef process fill:#EDE2C8,stroke:#9A7438,stroke-width:2px,color:#26352F
    classDef gate fill:#F3E2DD,stroke:#785650,stroke-width:2px,color:#3F2E2B
    classDef verified fill:#35594A,stroke:#26352F,stroke-width:2px,color:#FFFFFF

    class Data contract
    class Evaluation process
    class Gate gate
    class Presentation verified
```

1. **Board-level Stratified Partition**：依 PCB Board ID 嚴格隔離 Train／Test，避免同款 sibling images 跨 split 造成虛高表現。
2. **Paired Protocol & A100 Benchmarking**：Grouped 與 Leaky Control 共用單一 Board 08 的 30 張 final-test images；Seeds 42／43／44 下觀察到 **+21.3 pp** mAP50 差距。
3. **Hash-pinned ONNX Fidelity Gate**：aggregate fidelity 通過；`60/60` **Same-ONNX wrapper parity** 比較同一 ONNX artifact 的兩條執行路徑，**非 PyTorch reference**，不代表 PyTorch→ONNX per-box equivalence。
4. **NVIDIA L4 multi-backend benchmark**：60 張 calibration images 上，ONNX Runtime CUDA FP32 p50 為 **20.28 ms**；TensorRT FP16 通過 aggregate fidelity，但匯出後端未通過 frozen strict per-box parity gate。

---

## Fail-closed 工作站啟動時序

工作站啟動時先驗證 committed evidence 與 release contract，不會先下載 floating model。任何 evidence、contract、hash 或 runtime 失敗都會隱藏 inference controls，並保留可說明的 Recorded evidence 或 Degraded state。

```mermaid
%%{init: {'themeVariables': {'fontSize': '24px'}}}%%
flowchart TB
    accTitle: PCB 人工複核工作站的 fail-closed 啟動門控
    accDescr: 工作站先讀取 committed evidence 與 model contract；blocked contract 顯示 Recorded evidence，資料錯誤進入 Degraded，全部 release checks 通過才進入 Live。

    Start["啟動工作站 — 讀取 evidence + contract"]
    Gate{"Release checks 結果為何？"}
    Evidence["EVIDENCE — Recorded only"]
    Degraded["DEGRADED — inference disabled"]
    Live["LIVE — verified inference"]

    Start --> Gate
    Gate -->|blocked contract| Evidence
    Gate -->|missing · mismatch · failure| Degraded
    Gate -->|contract · hash · runtime passed| Live
    Evidence ~~~ Degraded ~~~ Live

    classDef neutral fill:#F3F0E8,stroke:#587069,stroke-width:2px,color:#26352F
    classDef decision fill:#EDE2C8,stroke:#9A7438,stroke-width:2px,color:#26352F
    classDef evidence fill:#DCE7DF,stroke:#35594A,stroke-width:2px,color:#26352F
    classDef blocked fill:#F3E2DD,stroke:#785650,stroke-width:2px,color:#3F2E2B
    classDef live fill:#35594A,stroke:#26352F,stroke-width:2px,color:#FFFFFF

    class Start neutral
    class Gate decision
    class Evidence evidence
    class Degraded blocked
    class Live live
```

> 目前 committed `model_contract.json` 為 blocked，因此公開作品集停在 **Recorded evidence mode**；上圖的 `LIVE` 是程式中的受控能力路徑，不是已上線服務。

---

## Deep Dive：實驗與部署 Pipeline

<details>
<summary>展開：PCB 板級防洩漏與成對實驗流程</summary>

這張圖展開 frozen Board-level split、Grouped／Leaky Control paired training，以及共同 Board 08 final test 上的受控差值評測。

```mermaid
%%{init: {'themeVariables': {'fontSize': '24px'}}}%%
flowchart TB
    accTitle: PCB 板級防洩漏與成對實驗流程
    accDescr: HRIPCB 依 Board ID 凍結分割，Grouped 與 Leaky Control 在三個 seeds 下共用 Board 08 final test，評估 same-board exposure effect。

    Dataset["HRIPCB Dataset — 10 PCB boards"]
    Split["Frozen Board Split — Board 08 held out"]
    Arms["Paired Arms — Grouped · Leaky"]
    Train["A100 Training — Seeds 42 · 43 · 44"]
    Test["Common Final Test — Board 08"]
    Effect["Leakage Effect — +21.3 pp mAP50"]

    Dataset --> Split --> Arms --> Train --> Test --> Effect

    classDef source fill:#DCE7DF,stroke:#35594A,stroke-width:2px,color:#26352F
    classDef process fill:#EDE2C8,stroke:#9A7438,stroke-width:2px,color:#26352F
    classDef result fill:#35594A,stroke:#26352F,stroke-width:2px,color:#FFFFFF

    class Dataset,Split source
    class Arms,Train,Test process
    class Effect result
```

- **Frozen split**：Grouped train 513 張且排除 Board 08；Leaky Control 以 30 張同板 sibling images 進行對照。
- **共同評測**：三個 seeds 共用 Board 08 final test；`Leaky - grouped: +21.3 pp`，paired F1 delta 為 `0.2546`。
- **統計邊界**：bootstrap resampling unit 是 image，不估計 between-board uncertainty。

</details>

<details>
<summary>展開：ONNX fidelity 與 NVIDIA L4 多後端部署管線</summary>

這張圖展開 PyTorch→ONNX fidelity、TensorRT build、NVIDIA L4 calibration latency 與 strict per-box parity gate。

```mermaid
%%{init: {'themeVariables': {'fontSize': '24px'}}}%%
flowchart TB
    accTitle: ONNX fidelity 與 NVIDIA L4 多後端部署管線
    accDescr: PyTorch 模型匯出 ONNX 後進行 wrapper parity、L4 多後端 latency 與 PyTorch-reference strict per-box parity；failed gate 只發布 calibration evidence。

    Source["PyTorch Checkpoint — Grouped Seed 42"]
    ONNX["ONNX Export — hash pinned"]
    Fidelity["Aggregate Fidelity — PASS"]
    Benchmark["NVIDIA L4 Benchmark — ORT · TensorRT"]
    Parity["Strict Per-box Parity — FAILED"]
    Evidence["Release Evidence — calibration only"]

    Source --> ONNX --> Fidelity --> Benchmark --> Parity --> Evidence

    classDef source fill:#DCE7DF,stroke:#35594A,stroke-width:2px,color:#26352F
    classDef process fill:#EDE2C8,stroke:#9A7438,stroke-width:2px,color:#26352F
    classDef pass fill:#35594A,stroke:#26352F,stroke-width:2px,color:#FFFFFF
    classDef blocked fill:#F3E2DD,stroke:#785650,stroke-width:2px,color:#3F2E2B

    class Source,ONNX source
    class Benchmark process
    class Fidelity pass
    class Parity,Evidence blocked
```

- **Aggregate fidelity**：Same-ONNX wrapper parity 為 `60/60`，但這不是 PyTorch reference equivalence。
- **L4 latency**：ONNX Runtime CUDA FP32 `p50: 20.28 ms · 49.32 FPS`；TensorRT FP16 p50 為 `51.12 ms`。
- **發布邊界**：PyTorch-reference strict per-box parity gate failed，因此只發布 calibration-only evidence，不宣稱 production SLA。

</details>

---

## 凍結實驗資料分割設計

| 資料集切分 (Partition) | 影像數量 (Images) | 板號防洩漏原則 (Board Policy) |
|---|---:|---|
| Common Final Test | 30 (5/class) | 鎖定 Board 08；兩組訓練集均排除此 30 張測試影像 |
| Validation | 60 (10/class) | 鎖定 Board 01；供超參數選擇與早停共用 |
| Calibration | 60 (10/class) | 鎖定 Board 01；獨立於驗證與訓練集，專供 ONNX/TensorRT 校準 |
| Grouped Train (無洩漏組) | 513 | 僅使用 Boards 04/05/06/07/09/10/11/12；絕無 Board 08 影像 |
| Leaky-control Train (洩漏對照組) | 513 | 替換 30 張同類別 Board 08 兄弟樣本 (Siblings) 進行對照 |

此分割一經凍結即不再變動，並以雜湊綁定；`configs/paired_protocol.yaml` 的 `frozen_hashes`
與 `reports/protocol/paired_split_manifest.json` 必須逐位元相符：

- Manifest SHA-256：`5996d595f5ce17fabd24e631ce580bbf9932a845f9898078267df8c2522892e5`
- Dataset SHA-256：`8e5f0c880af67019bfc7ab5b08a4e63cc33726c97b5a77a41ebb27ddb3709ed4`

---

## 實驗評測與硬體基準數據

### 1. 成對板級洩漏實驗結果 (A100 GPU, 3 Seeds)

在相同的單一 Board 08（30 張 final-test images）上，三種子平均評測結果呈現以下受控差異：

| 實驗組別 (Experiment Arm) | mAP50 (%) | mAP50-95 (%) | 證據邊界 |
|---|---:|---:|---|
| 嚴格分組組 (Grouped) | 63.30% ± 14.91% | 28.82% ± 6.54% | 單一 held-out Board 08 結果；不代表跨板母體或產線泛化 |
| 洩漏對照組 (Leaky Control) | 84.56% ± 3.75% | 40.08% ± 2.52% | same-board sibling exposure 對照組；在 frozen protocol 下高 +21.3 pp |

- **成對 F1 差值 (Paired F1 Delta)**：`0.2546`，paired image-bootstrap 95% 信賴區間為 `[0.2102, 0.3005]`。Resampling unit 是 image，不是 board；此區間不估計 between-board uncertainty。
- **機器可核對原值**（取自 `reports/paired_a100/final_metrics.json`，上表百分比即由此換算）：
  grouped mAP50 `0.6330 ± 0.1491`、leaky control mAP50 `0.8456 ± 0.0375`，差距 `21.3` 個百分點。

> **What this proves**：在這個 frozen dataset 與 training recipe 下，same-board sibling exposure 與單一 Board 08 final test 上的受控表現差異相對應。
>
> **What this does not prove**：此結果不建立跨 board 母體、新產品、factory-line 或 production generalization。

### 2. NVIDIA L4 多後端推論延遲評測

> **這是由 private package 衍生的 public metadata、calibration-only 證據，不是 production 效能宣稱。**
> 量測只在 60 張校準 (calibration) 影像上進行，跑在單一私有 L4 session；
> 本 repo **不發佈** TensorRT engine、不發佈 public model，也沒有任何 public checkpoint
> 對應這組數字。它記錄描述性 latency 與 aggregate calibration fidelity；strict per-box
> parity gate 未通過，因此不宣稱三個 backend 的 prediction sets 等價，也**不能**用來
> 推論產線 (production) 上的實際吞吐或良率。

在 60 張校準影像、batch 1、30 次 warmup、4 cycles（每個後端 240 次記錄）下，以
interleaved rotating backend order 實測：

| Backend | Precision | p50 (ms) | p95 (ms) | FPS from p50 |
|---|---|---:|---:|---:|
| PyTorch | FP32 | 60.86 | 62.36 | 16.43 |
| ONNX Runtime CUDA | FP32 | 20.28 | 20.87 | 49.32 |
| TensorRT | FP16 | 51.12 | 52.25 | 19.56 |

**ONNX Runtime CUDA FP32 是本次最快後端**。TensorRT FP16 相對 source checkpoint 的 mAP50-95 差值為 `-0.014537137094089408`，通過 `|Δ| < 0.02` 的 aggregate calibration fidelity gate。

- **機器可核對原值**（取自 `reports/benchmark_l4.json`，上表為其四捨五入）：
  TensorRT p50 `51.12191199998506` ms、p95 `52.25180029992771` ms；完整 720 筆 timing
  observations 位於 `reports/benchmark_l4_raw.json`。
- **strict per-box prediction-parity gate failed**：PyTorch reference 有 95 個 detections；
  ONNX Runtime CUDA 配對 57 個、漏配 reference/candidate 為 38/5，TensorRT 配對 56 個、
  漏配為 39/5。兩者皆有 40/60 images 未過 gate；完整 pseudonymized evidence 位於
  `reports/backend_parity_l4.json`。門檻在執行前已凍結，沒有事後放寬。

---

## PCB 六大檢測瑕疵類別

| 瑕疵類別 (Defect Class) | 瑕疵中文說明 | 產線常見成因與影響 |
|---|---|---|
| `missing_hole` | 漏孔 / 缺孔 | 鑽孔製程遺漏，導致後續元件無法插裝貫通 |
| `mouse_bite` | 鼠咬 / 邊緣缺口 | 蝕刻不均或板邊毛邊，可能造成線路阻抗異常 |
| `open_circuit` | 斷路 / 開路 | 銅箔線路斷裂，造成電氣信號無法導通中斷 |
| `short` | 短路 | 殘銅或錫橋相連，導致相鄰線路不正常導通 |
| `spur` | 針狀突出 / 殘銅 | 乾膜破損或蝕刻不全，易引發潛在短路風險 |
| `spurious_copper` | 雜銅 / 假銅斑 | 孤立無效銅渣，影響絕緣耐壓與高頻特性 |

---

## 快速開始

### 1. 安裝環境套件

```bash
# 建立虛擬環境並安裝依賴
uv sync --locked --no-editable

# 執行單元測試與程式碼風格稽核
uv run python -m pytest -v
uv run ruff check .
```

### 2. 資料集下載與防洩漏前處理

```bash
# 下載 HRIPCB 資料集並轉換為 YOLO 格式 (嚴格分組策略)
uv run python -m pcb_defect.data_prep.prepare --out data/pcb --strategy grouped --seed 42

# 驗證凍結分割協定並產生配對資料清單
uv run python -m pcb_defect.data_prep.paired \
  --source data/pcb \
  --config configs/paired_protocol.yaml \
  --artifacts reports/protocol \
  --runtime data/paired
```

### 3. GPU 實驗入口（先檢查 CLI，不會啟動訓練）

```bash
# 顯示所有必要參數；此命令不會下載 dataset、訓練或使用 GPU
uv run --locked --no-editable --extra train --group eval \
  python -m pcb_defect.experiment --help
```

完整、可續跑且帶有 input/hash gates 的流程已封裝於
[`notebooks/paired_experiment_a100.ipynb`](notebooks/paired_experiment_a100.ipynb) 與
[`notebooks/deployment_benchmark_l4.ipynb`](notebooks/deployment_benchmark_l4.ipynb)。Notebook
需要使用者明確提供 dataset 與 workspace 路徑；README 不提供會誤啟動訓練的裸命令。

---

## 專案結構

| 檔案 / 目錄 | 功能說明與職責 |
|---|---|
| `configs/paired_protocol.yaml` | 成對實驗資料分割與訓練超參數協定 |
| `configs/base_model.yaml` | YOLO26n 基礎模型下載與 SHA-256 驗證組態 |
| `src/pcb_defect/data_prep/paired.py` | HRIPCB 成對板級防洩漏資料準備 |
| `src/pcb_defect/experiment.py` | A100 訓練執行、斷點續跑門控與自動重試管理 |
| `src/pcb_defect/final_evaluation.py` | 單次一擊 (One-shot) 最終測試集評測 |
| `reports/protocol/` | 凍結分割 Manifest 與配對哈希驗證紀錄 |
| `reports/benchmark_l4.md` | Verified NVIDIA L4 latency、aggregate fidelity 與 failed strict parity 摘要 |

---

## 引用

目前軟體版本為 **v0.2.0**，已由 Zenodo 永久典藏：

- 版本 DOI：[`10.5281/zenodo.21912370`](https://doi.org/10.5281/zenodo.21912370)
- 全版本 DOI：[`10.5281/zenodo.21877496`](https://doi.org/10.5281/zenodo.21877496)

既有 **v0.1.0** source-and-metadata research package 仍以不可變版本 DOI 保存：

- 版本 DOI：[`10.5281/zenodo.21877497`](https://doi.org/10.5281/zenodo.21877497)

引用目前版本時請使用：

> kuotunyu. (2026). *PCB Defect Detection: Leakage-Aware Evaluation and Deployment Evidence* (Version 0.2.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.21912370

引用歷史 `v0.1.0` 時請使用：

> kuotunyu. (2026). *PCB Defect Detection: Leakage-Aware Evaluation and Deployment Evidence* (Version 0.1.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.21877497

機器可讀的引用資料請見 [`CITATION.cff`](CITATION.cff)。

---

## 授權與聲明

本專案原創程式碼採 [AGPL-3.0-or-later](LICENSE)。此 code license 不授予 HRIPCB 原始影像、標註、base weights、derived weights 或 exports 的再散佈權；詳見 [`docs/license-boundary.md`](docs/license-boundary.md)。
