# README 圖解可讀性重構設計

日期：2026-08-13

## 問題

GitHub 會將 Mermaid 產生的 SVG 等比例縮進 README 欄寬。現有圖解雖已設定 17–18 px 字體，但節點、participant 與長句過多，尤其六欄的 fail-closed Sequence Diagram 形成寬且高的畫布；整體縮放後，預設 100% 瀏覽器縮放下文字仍難以閱讀。只提高 `fontSize` 會同時放大畫布，無法可靠改善最終顯示大小。

## 決策

採用「降低資訊密度、分層揭露」方案：README 主敘事中的圖解必須在 GitHub 預設寬度即可閱讀；詳細技術路徑保留於 Deep Dive，不要求訪客先操作 Mermaid 的放大控制。

### 主圖可讀性 contract

- 主圖優先採用 `flowchart TB` 或緊湊的兩欄布局，避免橫跨四個以上主要視覺欄位。
- 每張主圖最多呈現 7 個核心節點；節點只保留辨識所需的短標題。
- 每個節點最多兩行，每行以約 24 個可見字元為上限；數值、限制與例外改放在圖下的 prose。
- Mermaid `themeVariables.fontSize` 最終使用 21 px；在 GitHub 預設顯示中兼顧辨識度與正文比例，高對比色與 `accTitle`／`accDescr` 繼續保留。
- 主圖必須讓讀者在 GitHub README、瀏覽器 100% 縮放下，不按 Mermaid zoom controls 也能讀出主要節點與箭頭。

### 圖解調整

1. **System Context**：改為由上而下的三層關係，合併 private data、GPU 與 artifacts，避免 trust-boundary 分支撐寬畫布。
2. **Evidence-first Architecture**：改為四個大型 stage 節點的垂直 pipeline；stage 細節移到緊接圖下的編號說明。
3. **Fail-closed Gate Flow**：以五個節點的狀態流程取代六 participant 的完整 Sequence Diagram，並垂直錯開 `EVIDENCE`、`DEGRADED` 與 `LIVE` 三種出口，避免三欄等寬縮放。
4. **Deep Dive**：保留兩張技術圖，但各 stage 改成垂直堆疊、減少長句；完整啟動互動改以文字步驟保留，不再讓高密度 Sequence Diagram 佔據 README 主畫面。

## 驗證

- Release-contract tests 驗證主圖節點數、方向、21 px 字級、可及性描述及禁止重新加入 `sequenceDiagram`。
- Mermaid CLI 必須能解析並渲染所有圖解。
- 在與 GitHub README 相近的 980–1100 px 內容寬度產生視覺預覽，確認文字不需放大即可辨識。
- 執行完整 pytest 與 Ruff，確認文件 contract 不影響其他專案行為。

## 非目標

- 不改變模型、評估數字、promotion gate 或公開證據邊界。
- 不加入外部圖解 SaaS、JavaScript 或額外 runtime dependency。
- 不以預先渲染圖片取代 Mermaid 原始碼。
