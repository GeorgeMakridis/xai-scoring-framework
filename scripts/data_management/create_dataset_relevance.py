#!/usr/bin/env python3
"""
Generate dataset_relevance.csv from metadata descriptions (TF-IDF + cosine similarity).
Use when domain-based relevance is insufficient (e.g., same domain but different sub-topics).
Supports two formats:
  - Option A: dataset_id_a, dataset_id_b, relevance (pairwise)
  - Option B: dataset_id, domain, relevance (dataset-to-domain)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
OUTPUT_PATH = os.path.join(DATA_ROOT, "shared", "dataset_relevance.csv")

# Minimum similarity threshold to include a pair
MIN_SIMILARITY = 0.3
# Max pairs per dataset to avoid huge file
MAX_PAIRS_PER_DATASET = 20


def get_all_datasets() -> pd.DataFrame:
    """Collect all datasets with metadata from all data types."""
    rows = []
    for dtype in ["tabular", "image", "text", "timeseries"]:
        meta_path = os.path.join(DATA_ROOT, dtype, "dataset_metadata.csv")
        if not os.path.exists(meta_path):
            continue
        try:
            df = pd.read_csv(meta_path, low_memory=False)
            desc_col = "description" if "description" in df.columns else ("description " if "description " in df.columns else None)
            if desc_col is None:
                df["description"] = ""
            else:
                df["description"] = df[desc_col]
            df = df[["dataset_id", "dataset_name", "domain", "description"]]
            df["data_type"] = dtype
            rows.append(df)
        except Exception as e:
            print(f"Warning: Could not read {meta_path}: {e}")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def compute_pairwise_relevance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise relevance from description TF-IDF cosine similarity."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("sklearn required for description-based relevance. Install: pip install scikit-learn")
        return pd.DataFrame()

    df = df.copy()
    df["description"] = df["description"].fillna("").astype(str)
    df["text"] = df["dataset_name"].fillna("") + " " + df["domain"].fillna("") + " " + df["description"]
    df["text"] = df["text"].str.lower().str[:5000]

    vectorizer = TfidfVectorizer(max_features=500, stop_words="english", min_df=1)
    X = vectorizer.fit_transform(df["text"])
    sim = cosine_similarity(X, X)

    pairs = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            s = sim[i, j]
            if s >= MIN_SIMILARITY:
                pairs.append((df.iloc[i]["dataset_id"], df.iloc[j]["dataset_id"], round(float(s), 4)))

    if not pairs:
        return pd.DataFrame()

    out = pd.DataFrame(pairs, columns=["dataset_id_a", "dataset_id_b", "relevance"])
    # Limit pairs per dataset
    counts = {}
    filtered = []
    for _, row in out.iterrows():
        a, b = str(row["dataset_id_a"]), str(row["dataset_id_b"])
        if counts.get(a, 0) < MAX_PAIRS_PER_DATASET and counts.get(b, 0) < MAX_PAIRS_PER_DATASET:
            filtered.append(row)
            counts[a] = counts.get(a, 0) + 1
            counts[b] = counts.get(b, 0) + 1
    return pd.DataFrame(filtered) if filtered else pd.DataFrame()


def main() -> int:
    df = get_all_datasets()
    if df.empty:
        print("No dataset metadata found.")
        return 1

    out_df = compute_pairwise_relevance(df)
    if out_df.empty:
        print("No pairwise relevance above threshold. Creating empty dataset_relevance.csv.")
        out_df = pd.DataFrame(columns=["dataset_id_a", "dataset_id_b", "relevance"])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(out_df)} rows to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
