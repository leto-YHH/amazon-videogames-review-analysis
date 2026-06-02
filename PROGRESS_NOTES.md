# 專題進度紀錄
## Amazon Video Games Review Analysis

---

## 專題目標

分析 Amazon Video Games 類別的消費者評論對產品銷售排名的影響，找出有影響力的評論者特徵，並用機器學習預測銷售排名。

---

## 目前完成進度

### ✅ 第一階段：資料準備
### ✅ 第二階段：探索性資料分析（EDA）
### ✅ 第三階段：影響力評論者分析
### ✅ 第四階段：BERTopic 主題建模
### ⬜ 第五階段：XGBoost 預測模型（下一步）
### ⬜ 第六階段：商業建議撰寫

---

## 資料集基本資訊

- **來源**：Amazon Reviews 2023（McAuley Lab, UCSD）
- **類別**：Video Games
- **商品數**：74,321 個
- **評論數**：3,670,127 則
- **評論者數**：2,312,291 人
- **時間範圍**：約 2000 年 ~ 2023 年
- **依變數**：Best Sellers Rank（Video Games 大類），取 log 轉換後使用

---

## 各 Notebook 說明

### `src/build_dataset.py` — 資料清理與合併

**做了什麼：**
- 讀取 metadata（商品資訊）和 review（評論）兩個原始 `.jsonl.gz` 檔案
- 只保留有 `Best Sellers Rank` 欄位的商品（64.4% 有此欄位）
- 只使用 `Video Games` 大類的 rank，捨棄子類別 rank
- 移除 rank 為 0 的異常值
- 對 rank 做 **log 轉換**（因為原始 rank 分佈極度右偏，log 轉換後比較適合建模）
- 移除沒有評論文字的評論
- 輸出三個 CSV 檔至 `data/processed/`

**輸出檔案：**
- `meta_clean.csv`：74,321 個商品的基本資訊
- `reviews_clean.csv`：3,670,127 則評論
- `dataset.csv`：評論與商品資訊合併後的完整資料集

---

### `notebooks/01_eda.ipynb` — 探索性資料分析

**做了什麼：**
分析資料的基本樣貌，為後續分析提供決策依據。

**產生的圖片：**

#### `01_rank_distribution.png`
- **左圖（藍色）**：原始 BSR 分佈，極度右偏，大多數商品集中在低排名（銷量差）區間
- **右圖（綠色）**：log(BSR) 分佈，轉換後分佈較均勻，確認 log 轉換有效
- **結論**：log_rank 是正確的建模目標

#### `02_review_volume_over_time.png`
- **折線面積圖**：每月評論數量的時間趨勢
- **關鍵發現**：
  - 2012 年後評論量大幅增加（Amazon Marketplace 擴張）
  - 2020 年有明顯峰值（疫情期間遊戲銷量爆發）
  - 2023 年後急速下降（資料集截止日期，非真實趨勢）

#### `03_rating_distribution.png`
- **柱狀圖**：各星評的評論數量與比例
- **關鍵發現**：
  - 典型 J 型分佈：61.9% 是五星，12.4% 是一星，中間的 2-4 星很少
  - 玩家評分行為：要嘛非常滿意，要嘛非常憤怒

#### `04_helpful_vote_distribution.png`
- **直方圖**：helpful_vote 分佈（只顯示 > 0 的部分，截斷至 200）
- **關鍵發現**：
  - 74.8% 的評論 helpful_vote = 0
  - 95% 的評論在 4 票以下
  - 99% 在 18 票以下
  - 最高有 10,369 票
  - **決策依據**：影響力評論者門檻設為 helpful_vote ≥ 10（前 2.1%）

#### `05_text_length_distribution.png`
- **直方圖**：評論字數分佈（截斷至 500 字）
- **關鍵發現**：
  - 中位數只有 24 字（大多數評論很短）
  - 但有長尾分佈，影響力評論者傾向寫長評論
  - 平均字數 56.9 字（被長評論拉高）

#### `06_verified_vs_unverified.png`
- **三格柱狀圖**：驗證購買 vs 非驗證購買評論的比較
- **關鍵發現（重要且反直覺）**：
  - 非驗證購買的平均 helpful_vote 是 3.27，遠高於驗證購買的 0.90
  - 非驗證購買的平均字數是 147.6 字，遠高於驗證購買的 42.0 字
  - **解釋**：非驗證購買者可能是媒體人、KOL 或重度玩家，雖然沒有購買紀錄，但寫出的評論更有深度，也更受其他人認可
  - **商業意涵**：不應該只看驗證購買的評論，非驗證購買的評論反而可能更有影響力

#### `07_correlation_with_logrank.png`
- **水平柱狀圖**：各數值特徵與 log_rank 的 Pearson 相關係數
- **紅色 = 正相關**（特徵越高，rank 數字越大 = 銷量越差）
- **藍色 = 負相關**（特徵越高，rank 數字越小 = 銷量越好）
- **關鍵發現**：
  - `avg_rating`（-0.29）：平均評分越高，排名越好，相關性最強
  - `total_reviews`（-0.27）：評論數越多，排名越好
  - `verified_ratio`（-0.12）：驗證購買比例越高，排名越好
  - `avg_text_length`（+0.08）：評論越長的商品排名反而越差（可能因為差評者寫得比較長）
  - `avg_helpful_vote`（+0.008）：在商品層級幾乎無相關，但在評論者層級分析更有意義

---

### `notebooks/02_influential_reviewers.ipynb` — 影響力評論者分析

**做了什麼：**
定義並分析影響力評論者的特徵，以及他們對商品排名的影響。

**定義：** helpful_vote ≥ 10 的評論者 = 影響力評論者

**主要數字：**
- 影響力評論者：63,816 人（佔所有評論者 2.8%）
- 一般評論者：2,248,475 人

**產生的圖片：**

#### `08_influential_vs_regular.png`
- **六格柱狀圖**：影響力評論者 vs 一般評論者的各項特徵比較
- **關鍵發現**：
  - 評論字數：181.6 字 vs 42.9 字（影響力評論者寫 4 倍多）
  - Helpful Vote：22.0 vs 0.4（影響力評論者高出 55 倍）
  - 驗證購買比例：64.8% vs 89.2%（影響力評論者驗證購買比例更低）
  - 星評：3.5 vs 4.0（影響力評論者評分標準更嚴格）
  - **影響力評論者畫像**：重度玩家或媒體人，寫深度長評、評分嚴格、不一定每次都購買

#### `09_review_patterns.png`
- **左圖**：評論字數分佈比較（影響力 vs 一般）— 影響力評論者的字數分佈明顯更向右延伸
- **右圖**：星評分佈比較 — 影響力評論者在 1-4 星的比例都略高，5 星比例較低（50% vs 62%），確認他們評分更嚴格

#### `10_influential_vs_rank.png`
- **左圖散點圖**：影響力評論比例 vs log(Rank)，r = 0.085，關係微弱
- **右圖柱狀圖**：按影響力評論比例分組的中位數排名
- **關鍵發現（最重要）**：
  - 沒有影響力評論：中位數 rank 92,121
  - 少量影響力評論（0~10%）：rank 41,762（最好！）
  - 中量影響力評論（10~30%）：rank 76,301
  - 高比例影響力評論（>30%）：rank 122,256（最差）
  - **倒 U 型關係**：少量影響力評論最有利，但太多反而更差
  - **可能解釋**：影響力評論者評分嚴格，比例太高會拉低商品評分，反而傷害排名

---

### `notebooks/03_bertopic.ipynb` — BERTopic 主題建模

**做了什麼：**
從評論文字中萃取主題，找出哪些主題的商品銷量更好。

**技術流程：**
1. 篩選 helpful_vote ≥ 1 且字數 ≥ 10 的高品質評論（835,227 則）
2. 從中隨機抽樣 100,000 則
3. 用 BERTopic 建模：BERT 向量化 → UMAP 降維 → HDBSCAN 分群
4. 找出 49 個主題
5. 分析各主題的商品中位數排名

**產生的圖片：**

#### `11_topic_distribution.png`
- **水平柱狀圖**：前 20 個主題的評論數量
- **關鍵發現**：
  - Topic 0（game, br, like）最大，有 9,513 則，是一般遊戲討論
  - Topic 1（headset, sound, mic）6,970 則，耳機評論
  - Topic 2（keyboard, keys, key）4,137 則，鍵盤評論
  - 遊戲周邊配件（耳機、鍵盤、滑鼠、控制器）佔了大量主題
  - 也有特定遊戲主題：wii/mario/zelda、pokemon、kingdom hearts、madden

#### `12_topic_vs_rank.png`
- **水平柱狀圖**：各主題的中位數銷售排名（綠色=排名好，紅色=排名差）
- **關鍵發現**：
  - **排名最好的主題**：stand/sturdy/stands（支架）、dock/switch/charger（充電配件）、love/great/easy（正面情感）、mouse/buttons/mice（滑鼠）
  - **排名最差的主題**：pokemon/pokmon/game（Pokemon 遊戲）、buttons/button/working（按鍵故障投訴）、cable/tv/hdmi（連接線）
  - **核心洞察**：周邊配件類商品的銷售排名普遍優於遊戲軟體本身，消費者購買配件更依賴評論，購買遊戲更依賴品牌和口碑
  - **負面主題傷害排名**：討論「按鍵故障」的評論與差排名高度相關，代表品質問題評論對銷售有明顯負面影響

---

## 下一步：XGBoost 預測模型

**目標：** 把前面分析出來的所有特徵整合，訓練一個預測商品排名的模型。

**輸入特徵：**
- 來自 EDA：avg_rating、total_reviews、verified_ratio、avg_helpful_vote、avg_text_length
- 來自影響力評論者分析：inf_review_ratio、inf_review_count
- 來自 BERTopic：各主題的評論比例（product_topic_features.csv）
- 來自 metadata：price、date_available

**預測目標：** log_rank

**評估指標：** R²、RMSE、特徵重要性圖

---

## 專案結構

```
amazon-videogames-review-analysis/
├── data/
│   ├── raw/              ← 原始 .jsonl.gz 檔案（未上傳 GitHub）
│   └── processed/        ← 清理後的 CSV 檔（未上傳 GitHub）
│       ├── meta_clean.csv
│       ├── reviews_clean.csv
│       ├── dataset.csv
│       └── product_topic_features.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_influential_reviewers.ipynb
│   └── 03_bertopic.ipynb
├── src/
│   └── build_dataset.py
├── outputs/              ← 所有圖片（已上傳 GitHub）
│   ├── 01_rank_distribution.png
│   ├── 02_review_volume_over_time.png
│   ├── 03_rating_distribution.png
│   ├── 04_helpful_vote_distribution.png
│   ├── 05_text_length_distribution.png
│   ├── 06_verified_vs_unverified.png
│   ├── 07_correlation_with_logrank.png
│   ├── 08_influential_vs_regular.png
│   ├── 09_review_patterns.png
│   ├── 10_influential_vs_rank.png
│   ├── 11_topic_distribution.png
│   └── 12_topic_vs_rank.png
├── .gitignore
├── requirements.txt
└── README.md
```
