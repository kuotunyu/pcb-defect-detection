# PCB 人工複核工作站 UI／UX 設計規格

日期：2026-08-13  
狀態：已確認設計方向，待實作計畫

## 1. 目標

首要目標不是建構正式工廠系統，而是讓 GitHub 訪客在 30 秒內感受到作者具備完整的 Computer Vision Engineer 能力：不只會訓練模型，還理解資料洩漏、評估設計、backend fidelity、deployment gate、發布邊界與產品化呈現。

成功條件：

- 第一屏即可回答「這個專案做什麼、成果多強、為什麼可信」。
- UI 看起來像 PCB 人工複核工作站，而不是 Gradio 預設表單或錯誤告示。
- 正體中文（`zh-TW`）為主要語言；技術專有名詞保留原文。
- 內文以 17–18 px 為基準，不以縮小文字換取資訊密度。
- 減少無效空白與巢狀層級，桌機畫面善用可用寬度。
- 使用低飽和莫蘭迪色彩，具設計感但不炫技。
- 不誇大模型能力，不把示意影像宣稱成真實 inference result。
- README 必須直接展示高品質桌機 screenshot；訪客不需啟動專案也能看到成果。

## 2. 已考慮的方案

### 方案 A：只美化現有 Gradio

保留單檔 `app.py`，加入 custom CSS 與較好的排版。成本最低，但畫面仍容易呈現典型 Gradio demo 感，狀態、證據與 inference 邏輯也會繼續耦合。

### 方案 B：獨立 React／Next.js 前端與 Python backend

視覺自由度最高，但會引入 Node toolchain、API、雙套測試與部署工作。以本專案的作品集目標來說，額外複雜度無法轉換成同比例的說服力。

### 方案 C：模組化 Gradio 工作站＋公開 screenshot（採用）

保留 Python-first 與一行啟動的優勢，將 presentation、evidence、model state、inference 拆成清楚模組，使用 Gradio Blocks、Theme、custom CSS 與少量 HTML 建立完整作品集頁。另以固定 viewport 產出桌機與手機 screenshot，嵌入 README。

此方案融合三種敘事優點：

- 影像複核畫布負責第一眼的產品感。
- 精選 KPI 與 evidence card 證明工程深度。
- Promotion Gate 持續保持 fail-closed，不因展示需求而繞過發布合約。

## 3. 核心敘事與頁面順序

頁面採單頁式資訊架構，導覽列只提供 anchor navigation，不建立不必要的多頁層級。

### 3.1 Header

- 品牌：`PCB Defect Intelligence`
- 導覽：`工作站`、`模型證據`、`瑕疵類別`
- GitHub repository 外部連結
- 桌機保持單列；手機只保留品牌標誌與 GitHub CTA，避免狹窄選單擠壓內容。

### 3.2 Hero

主標：

> 從資料切分到 Deployment Gate
> 完整呈現 PCB 瑕疵偵測工程

副標需交代：YOLO26n、六類裸板瑕疵、Board-level protocol、資料洩漏效應與 evidence-driven release boundary。

Hero 右側顯示工作站視覺預覽。第一屏同時具備：

- 原始影像／標註結果並排的產品形象。
- Model family、supported classes 與 public inference 狀態。
- Promotion Gate 狀態。
- 明確模式標籤，避免把介面示意誤認成 live inference。

CTA 僅保留兩個：`查看複核工作站`、`瀏覽工程證據`。

### 3.3 KPI Strip

只顯示四個高訊號指標：

- `6 類` Defect Classes
- `+21.3 pp` Leakage Effect
- `63.30%` Grouped mAP50（三 seeds 平均）
- `20.28 ms` ONNX Runtime CUDA FP32 p50（NVIDIA L4 calibration-only）

每個 KPI 都附短 context，避免把單一數字脫離實驗條件呈現。

### 3.4 人工複核工作站

桌機採主視覺與狀態摘要雙欄：

- 左欄：原始影像／標註結果並排的 synthetic interface illustration。
- 右欄：Promotion Gate、Mode、Model family、Supported classes 與 Public inference。
- 所有 illustration 都標示為非模型輸出，不呈現虛構 confidence 或座標。

手機改為單欄依序呈現，不等比例縮小桌機版。

工作站依啟動狀態呈現不同能力，詳見第 6 節。

### 3.5 工程證據

以三張主卡建立故事線：

1. `Board-level Split`：說明為何要用 Board ID 分割。
2. `Paired Evaluation`：呈現 grouped 與 leaky-control 的 frozen paired protocol。
3. `Promotion Gate`：aggregate fidelity passed，但 strict per-box parity failed，因此阻擋發布。

每張卡直接連結 committed evidence，而不是只有摘要文字。進階 latency、per-class metrics 與限制條件採 disclosure／accordion 展開，避免首頁第一層承載所有研究細節。

### 3.6 六種瑕疵

使用 2×3 desktop grid、單欄 mobile list 呈現：

- `missing_hole`
- `mouse_bite`
- `open_circuit`
- `short`
- `spur`
- `spurious_copper`

每張卡只放英文 class name、正體中文名稱與一句判讀描述。若沒有合法可發布的原始影像，不使用 HRIPCB 圖片縮圖。

### 3.7 Footer／Repository CTA

用一個低層級區塊收尾，連結 Repository、Model Card 與 Research Package；Data Card 與 License Boundary 由 Research Package 繼續導覽。不保留 Gradio 預設 footer 作為主要視覺元素。

## 4. 視覺系統

設計語彙為 `Calm precision`：安靜、精密、可信。PCB 影像與 evidence 是主角，介面負責建立秩序。

### 4.1 色彩

- Pine `#3F5D4D`：主要操作、品牌與深色工作站背景。
- Sage `#738E7D`：次要重點、verified／pass 狀態。
- Warm Ivory `#F4F3EF`：頁面底色，降低純白刺眼感。
- Paper `#FFFEFB`：卡片底色。
- Peach `#E8C5B8`：bounding box、注意提示與 recorded mode。
- Brick `#9D6257`：blocked／error；保持低飽和但需符合對比要求。
- Ink `#26332E`：主文字。
- Muted `#63706A`：次要文字。

色彩不可單獨承擔狀態語意；每個狀態同時包含 icon、文字與 label。

### 4.2 Typography

字體 stack：`Inter, "Noto Sans TC", "Microsoft JhengHei", system-ui, sans-serif`。

- Display：44–60 px，mobile 34–40 px。
- Section heading：28–36 px。
- Card heading：18–22 px。
- Body：17–18 px，line-height 1.6–1.7。
- Label／metadata：13–14 px；不得以 10–11 px 作為主要資訊。
- 數字採 tabular numerals，方便比較 metrics。

### 4.3 Spacing 與層級

- 全站僅保留 Page、Section、Card 三層。
- Section vertical spacing：32–48 px。
- Card gap：12–16 px。
- Card padding：18–24 px。
- Border radius：12–20 px。
- 不在 Card 中再次堆疊具有完整邊框與陰影的 Card。
- 最大內容寬度約 1280–1360 px；避免在寬螢幕中央留下大量無效空白。

### 4.4 Motion

- 互動轉場 160–220 ms。
- 只使用 hover elevation、focus highlight、metric fade-in 與 detection focus。
- 遵循 `prefers-reduced-motion`。
- 不使用霓虹、漂浮粒子、3D tilt 或持續循環動畫。

## 5. 技術架構

保留 Gradio 6.19 與 Python 3.11。`app.py` 僅負責組裝與啟動，避免繼續成長為同時處理所有責任的單檔。

建議模組：

- `app/app.py`：建立 runtime state、組裝 UI、`launch()`。
- `app/evidence.py`：讀取及驗證 committed JSON evidence，輸出 UI-ready summary。
- `app/inference.py`：model resolution、hash verification、ONNX session、preprocess／postprocess。
- `app/models.py`：`AppMode`、`AppState`、`Metric`、`Detection` 等 dataclass／enum。
- `app/ui.py`：Gradio Blocks、event wiring 與 view composition。
- `app/theme.py`：Theme、CSS、固定 copy 與色彩 token。
- `app/assets/`：只放專案自有的 UI illustration、icon 或合法可發布素材。

資料流：

1. 啟動時讀取 `model_contract.json` 與 committed evidence。
2. `build_app_state()` 先驗證 evidence，再判定 `LIVE`、`EVIDENCE` 或 `DEGRADED`。
3. UI 無論何種狀態都完整 render；只有 inference capability 依 state 解鎖。
4. `LIVE` 模式下先 hash-verify ONNX，再建立 session。
5. inference handler 只接收已初始化的 session；不得自行下載 floating revision 或繞過 contract。

不加入帳號、資料庫、多人協作、批次任務佇列或真實複核寫回。這些功能對作品集首要目標沒有足夠投資報酬率。

## 6. 模式與狀態設計

### 6.1 `EVIDENCE`（目前預設）

- 完整顯示 Hero、KPI、workstation shell、evidence 與 defect taxonomy。
- 上傳與 `Run inference` 不出現或保持明確 disabled。
- Promotion Gate 以緊湊狀態卡呈現，不再取代整頁內容。
- 介面用的 PCB 視覺必須是專案自有 illustration，並標示 `介面示意 · 非模型輸出`。
- 真實 metrics 僅來自 committed reports，禁止虛構 per-box confidence 或 latency。

### 6.2 `LIVE`

只有當 contract 為 `passed`、ONNX hash 正確且 session 成功建立時啟用：

- 顯示圖片上傳、confidence slider 與執行按鈕。
- 原圖／偵測結果並排。
- 顯示 per-box class、confidence、coordinates 與當次 end-to-end latency。
- 保留 model hash／revision 與 gate status。

### 6.3 `DEGRADED`

Evidence report 缺失、schema 無效或 model load error 時：

- 頁面仍可閱讀，不因單一資料失敗整頁 crash。
- 受影響區塊顯示 inline error 與 evidence path。
- 不以預設值冒充真實 metrics。
- 若 contract 宣告 passed 但 session 建立失敗，立即回退至 `DEGRADED`，不得改用未驗證模型。

### 6.4 空白與 inference error

- 尚未選擇圖片：顯示檔案要求與工作流程，不顯示大面積空盒。
- inference failure：保留使用者輸入、在工作站內顯示錯誤，允許重試。
- 不支援的檔案：在選取階段顯示正體中文提示。
- 未偵測到瑕疵：使用中性 `未偵測到高於目前 confidence threshold 的瑕疵`，不可宣稱 PCB 無缺陷。

## 7. 資產與誠信邊界

HRIPCB 圖片與 annotations 的上游授權尚未確認，因此：

- 不將六張 demo image、GIF、prediction grid 或 SAHI comparison 重新加入 public repository。
- 首頁工作站預覽使用專案自行建立的 synthetic／illustrative PCB artwork。
- illustration 必須明確標示非模型輸出，不放虛構 confidence。
- 所有 KPI 都由 committed JSON 解析，不在 UI 中手動維護第二份數字。
- `20.28 ms` 必須標示 ONNX Runtime CUDA FP32、NVIDIA L4、calibration-only，不能包裝為 production SLA。
- `+21.3 pp` 必須標示為 frozen paired protocol 下的 same-board sibling exposure effect，而非普遍模型結論。

## 8. README 與 GitHub 第一印象

UI 完成後產出：

- `docs/assets/ui-workstation-desktop.png`：README 主視覺，固定 1440×900。
- `docs/assets/ui-workstation-mobile.png`：響應式設計證據，固定 390×844。

README 頂部順序調整為：

1. badges 與一句價值主張。
2. desktop UI screenshot。
3. 三個重點：leakage-aware protocol、paired evaluation、fail-closed deployment gate。
4. Quick start。
5. 深入 evidence 與研究文件。

不使用 autoplay GIF；靜態 screenshot 更清晰、下載更快，也避開未授權 dataset media。

## 9. Accessibility 與響應式

- 文字與背景達 WCAG AA 對比。
- 所有互動元件具有 keyboard focus state。
- 圖示不可取代文字 label。
- 桌機 1100 px 以上顯示雙欄工作站。
- 768–1099 px 縮為較緊湊雙欄或單欄，依實際可讀性決定。
- 767 px 以下固定單欄；先結果影像，再摘要與狀態。
- 不使用橫向捲動承載必要資訊。
- Dataframe 在手機改為 compact cards 或可讀的直向列，不縮成小字表格。

## 10. 驗證與測試

### 10.1 Unit tests

- evidence parser 正確映射 `final_metrics.json`、`benchmark_l4.json` 與 parity status。
- 缺檔、invalid JSON、schema mismatch 進入 `DEGRADED`。
- blocked contract 不建立 session、不觸發 network download。
- passed contract 必須通過 SHA-256 才能進入 `LIVE`。
- detection table 與 annotations 仍使用 canonical class order。

### 10.2 UI smoke tests

- app import 與 Blocks 建立成功。
- 三種 `AppMode` 都能 render。
- `EVIDENCE` 不顯示可執行 inference controls。
- `LIVE` 才顯示 upload、confidence 與 run action。
- 所有主要正體中文標題、mode label、限制條件存在。

### 10.3 Browser verification

- desktop 1440×900、tablet 1024×768、mobile 390×844。
- 檢查無水平 overflow、文字截斷與不可讀小字。
- 檢查 keyboard navigation、focus、reduced motion 與 disabled state。
- 擷取 desktop／mobile screenshot 並人工檢視。

### 10.4 Repository verification

- `pytest` 與 `ruff` 通過。
- README screenshot path 有效。
- repository 不包含 HRIPCB raw image、checkpoint、ONNX 或 TensorRT engine。
- UI 顯示的數字與 committed evidence 一致。

## 11. 完成定義

- 使用者進入頁面不再首先看到全頁 `Deployment blocked`。
- 第一個 viewport 同時呈現產品用途、工程價值與誠信狀態。
- 桌機與手機均舒適可讀，無主要文字小於 13 px、內文不小於 17 px。
- blocked model 仍能提供完整、可信且有設計感的作品集體驗。
- README 直接展示 UI screenshot。
- 未新增任何無法由 committed evidence 或明確 illustration label 支撐的模型宣稱。
