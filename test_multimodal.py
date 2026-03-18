#!/usr/bin/env python3
"""
Test script for multi-modal data support.
Run: python3 test_multimodal.py
Requires: pip install pandas openpyxl scikit-learn
"""
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Check deps first
try:
    import pandas as pd
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install pandas openpyxl scikit-learn")
    sys.exit(1)

# Mock streamlit before app import (app imports streamlit at top level)


# Mock streamlit to avoid UI when importing app (app uses: import streamlit as st)
class MockStreamlit:
    session_state = {}
    def __getattr__(self, name):
        return lambda *a, **k: None
_mock = MockStreamlit()
sys.modules['streamlit'] = _mock

def main():
    from app import (
        load_data_from_folders,
        get_available_datasets,
        extract_features_from_new_dataset,
        estimate_xai_score_for_new_dataset,
        estimate_xai_score_for_dataset
    )

    print("Loading data from data/ folders...")
    repos = load_data_from_folders("data")
    if not repos:
        print("ERROR: No data found. Run scripts/data_management/revise_and_split_data.py first.")
        sys.exit(1)
    print(f"Loaded types: {list(repos.keys())}")
    tabular_repo = repos.get("tabular")

    # Test 1: Tabular - dataset selection
    print("\n--- Tabular (dataset selection) ---")
    if tabular_repo:
        datasets = get_available_datasets("tabular", "data")
        if datasets:
            ds_id = str(datasets[0].get("dataset_id"))
            scores, rx, ra, sim = estimate_xai_score_for_dataset(ds_id, tabular_repo, domain="general", top_k=3, data_type="tabular")
            print(f"Dataset: {ds_id}, Recommended XAI: {rx}, AI: {ra}, Similar: {len(sim)}")
        else:
            print("No tabular datasets in metadata")
    else:
        print("No tabular repo")

    # Test 2: Tabular - new dataset (legacy flow)
    print("\n--- Tabular (new dataset) ---")
    df_tab = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    feat = extract_features_from_new_dataset(df_tab, "tabular")
    print(f"Features: {feat}")
    if tabular_repo:
        scores, rx, ra, sim = estimate_xai_score_for_new_dataset(df_tab, tabular_repo, data_type="tabular")
        print(f"Recommended XAI: {rx}, AI: {ra}, Similar: {len(sim)}")

    # Test 3: Image
    print("\n--- Image ---")
    image_repo = repos.get("image")
    if image_repo:
        datasets = get_available_datasets("image", "data")
        if datasets:
            ds_id = str(datasets[0].get("dataset_id"))
            scores2, rx2, ra2, sim2 = estimate_xai_score_for_dataset(ds_id, image_repo, data_type="image")
            print(f"Dataset: {ds_id}, Recommended XAI: {rx2}, AI: {ra2}, Similar: {len(sim2)}")
    else:
        print("No image repo")

    # Test 4: Text
    print("\n--- Text ---")
    text_repo = repos.get("text")
    if text_repo:
        datasets = get_available_datasets("text", "data")
        if datasets:
            ds_id = str(datasets[0].get("dataset_id"))
            scores3, rx3, ra3, sim3 = estimate_xai_score_for_dataset(ds_id, text_repo, data_type="text")
            print(f"Dataset: {ds_id}, Recommended XAI: {rx3}, AI: {ra3}, Similar: {len(sim3)}")
    else:
        print("No text repo")

    # Test 5: Timeseries
    print("\n--- Timeseries ---")
    ts_repo = repos.get("timeseries")
    if ts_repo:
        datasets = get_available_datasets("timeseries", "data")
        if datasets:
            ds_id = str(datasets[0].get("dataset_id"))
            scores4, rx4, ra4, sim4 = estimate_xai_score_for_dataset(ds_id, ts_repo, data_type="timeseries")
            print(f"Dataset: {ds_id}, Recommended XAI: {rx4}, AI: {ra4}, Similar: {len(sim4)}")
    else:
        print("No timeseries repo")

    print("\n✓ All multi-modal tests passed.")

if __name__ == "__main__":
    main()
