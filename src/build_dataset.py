"""
Amazon Reviews 2023 - Video Games
資料清理與合併腳本

執行前請先下載兩個檔案（放在同一個資料夾）：
  curl -O https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Video_Games.jsonl.gz
  curl -O https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Video_Games.jsonl.gz

執行：
  pip install pandas
  python build_dataset.py

輸出：
  meta_clean.csv     清理後的商品資料
  reviews_clean.csv  清理後的評論資料
  dataset.csv        合併後的分析用資料集（每列 = 一則評論 + 該商品的特徵）
"""

import gzip
import json
import re
import pandas as pd
from pathlib import Path

META_FILE   = r"C:\Users\PC\Desktop\amazon-videogames-review-analysis\data\raw\meta_Video_Games.jsonl.gz"
REVIEW_FILE = r"C:\Users\PC\Desktop\amazon-videogames-review-analysis\data\raw\Video_Games.jsonl.gz"

# ── 1. 讀取 metadata ──────────────────────────────────────────────────────

print("【1/4】讀取 metadata...")

meta_rows = []

with gzip.open(META_FILE, "rt", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line.strip())
        details = item.get("details", {})

        # 只保留有 Video Games 大類 rank 的商品
        bsr = details.get("Best Sellers Rank", {})
        rank = bsr.get("Video Games") if isinstance(bsr, dict) else None
        if rank is None:
            continue

        # 取得上市日期（優先用 Date First Available，沒有的話用 Release date）
        date_str = details.get("Date First Available") or details.get("Release date")

        meta_rows.append({
            "parent_asin"    : item.get("parent_asin"),
            "title"          : item.get("title"),
            "price"          : item.get("price"),
            "average_rating" : item.get("average_rating"),
            "rating_number"  : item.get("rating_number"),
            "rank"           : rank,
            "log_rank"       : None,          # 後面計算
            "store"          : item.get("store"),
            "date_available" : date_str,
        })

meta_df = pd.DataFrame(meta_rows)
print(f"  原始有 rank 的商品數：{len(meta_df)}")

# 移除 rank 異常值（rank = 0 或極大值通常是資料錯誤）
meta_df = meta_df[meta_df["rank"] > 0]

# log 轉換
import numpy as np
meta_df["log_rank"] = np.log(meta_df["rank"])

# 移除沒有 parent_asin 的
meta_df = meta_df.dropna(subset=["parent_asin"])
meta_df = meta_df.drop_duplicates(subset=["parent_asin"])

print(f"  清理後商品數：{len(meta_df)}")
print(f"  rank 範圍：{meta_df['rank'].min()} ~ {meta_df['rank'].max()}")
print(f"  有價格的商品：{meta_df['price'].notna().sum()} 筆")
print()

# ── 2. 讀取評論 ───────────────────────────────────────────────────────────

print("【2/4】讀取評論（檔案較大，請稍候）...")

valid_asins = set(meta_df["parent_asin"].tolist())
review_rows = []

with gzip.open(REVIEW_FILE, "rt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i % 500000 == 0 and i > 0:
            print(f"  已讀取 {i:,} 筆評論...")

        rev = json.loads(line.strip())

        # 只保留有對應 metadata 的商品
        asin = rev.get("parent_asin")
        if asin not in valid_asins:
            continue

        review_rows.append({
            "parent_asin"      : asin,
            "user_id"          : rev.get("user_id"),
            "rating"           : rev.get("rating"),
            "title"            : rev.get("title"),
            "text"             : rev.get("text"),
            "helpful_vote"     : rev.get("helpful_vote", 0),
            "verified_purchase": rev.get("verified_purchase", False),
            "timestamp"        : rev.get("timestamp"),
        })

review_df = pd.DataFrame(review_rows)
print(f"  原始評論數：{len(review_df):,}")

# 移除沒有評論文字的
review_df = review_df.dropna(subset=["text"])
review_df = review_df[review_df["text"].str.strip() != ""]

# 轉換 timestamp 為日期
review_df["date"] = pd.to_datetime(review_df["timestamp"], unit="ms", errors="coerce")

# 計算評論字數
review_df["text_length"] = review_df["text"].str.split().str.len()

print(f"  清理後評論數：{len(review_df):,}")
print(f"  涵蓋商品數：{review_df['parent_asin'].nunique():,}")
print(f"  涵蓋評論者數：{review_df['user_id'].nunique():,}")
print(f"  驗證購買比例：{review_df['verified_purchase'].mean()*100:.1f}%")
print(f"  平均評論字數：{review_df['text_length'].mean():.1f} 字")
print()

# ── 3. 合併 ───────────────────────────────────────────────────────────────

print("【3/4】合併 metadata 與評論...")

dataset = review_df.merge(
    meta_df[["parent_asin", "rank", "log_rank", "price", "average_rating", "rating_number", "store"]],
    on="parent_asin",
    how="inner"
)

print(f"  合併後總筆數：{len(dataset):,}")
print()

# ── 4. 儲存 ───────────────────────────────────────────────────────────────

print("【4/4】儲存檔案...")

meta_df.to_csv("meta_clean.csv", index=False, encoding="utf-8-sig")
review_df.to_csv("reviews_clean.csv", index=False, encoding="utf-8-sig")
dataset.to_csv("dataset.csv", index=False, encoding="utf-8-sig")

print("  ✅ meta_clean.csv")
print("  ✅ reviews_clean.csv")
print("  ✅ dataset.csv")
print()

# ── 5. 基本統計摘要 ───────────────────────────────────────────────────────

print("=" * 50)
print("資料集摘要")
print("=" * 50)
print(f"商品數        : {dataset['parent_asin'].nunique():,}")
print(f"評論數        : {len(dataset):,}")
print(f"評論者數      : {dataset['user_id'].nunique():,}")
print(f"平均每商品評論: {len(dataset)/dataset['parent_asin'].nunique():.1f} 則")
print()
print("rank 分佈（分位數）：")
print(dataset["rank"].describe(percentiles=[.25,.5,.75,.95]).to_string())
print()
print("log_rank 分佈：")
print(dataset["log_rank"].describe().to_string())