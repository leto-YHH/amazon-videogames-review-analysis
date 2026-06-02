# Amazon Video Games Review Analysis

分析 Amazon Video Games 類別的消費者評論對產品銷售排名的影響。

## 專題目標

- 利用 **BERTopic** 從評論文字中萃取主題特徵
- 找出具有影響力的評論者，並分析其共同特徵
- 使用 **XGBoost** 預測產品銷售排名（Best Sellers Rank）
- 提供基於分析結果的商業建議

## 資料來源

[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) — McAuley Lab, UCSD  
類別：Video Games｜評論數：~4.6M｜商品數：~137K

> 原始資料因檔案過大未上傳，請依下方步驟自行下載。

## 專案結構

```
├── data/
│   ├── raw/          # 原始 .jsonl.gz 檔案（需自行下載）
│   └── processed/    # 清理後的 .csv 檔案（由腳本產生）
├── notebooks/        # 分析用 Jupyter Notebook
├── src/              # Python 腳本
│   └── build_dataset.py
├── outputs/          # 圖表與結果
├── requirements.txt
└── README.md
```

## 使用方式

### 1. 安裝套件

```bash
pip install -r requirements.txt
```

### 2. 下載原始資料

```bash
cd data/raw
curl -O https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Video_Games.jsonl.gz
curl -O https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Video_Games.jsonl.gz
```

### 3. 建立資料集

```bash
python src/build_dataset.py
```

輸出至 `data/processed/`：
- `meta_clean.csv` — 清理後的商品資料
- `reviews_clean.csv` — 清理後的評論資料
- `dataset.csv` — 合併後的分析用資料集

## 分析流程

| 步驟 | 內容 | 方法 |
|------|------|------|
| 1 | 資料清理與 EDA | pandas、matplotlib |
| 2 | 評論文字主題建模 | BERTopic |
| 3 | 影響力評論者識別 | helpful_vote、統計分析 |
| 4 | 銷售排名預測 | XGBoost |
| 5 | 商業建議 | 基於特徵重要性 |

## 參考文獻

Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). Bridging Language and Items for Retrieval and Recommendation. *arXiv preprint arXiv:2403.03952*.
