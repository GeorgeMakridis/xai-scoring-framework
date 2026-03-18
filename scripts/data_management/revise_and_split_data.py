#!/usr/bin/env python3
"""
Revise and split data by type: fix Simplicity, Precision, apply deterministic estimates,
and write to data/{tabular,image,text,timeseries}/ with CSV files.
"""

import os
import sys
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXCEL_PATH = os.path.join(PROJECT_ROOT, "Fame XAI scoring Framework_v2-2.xlsx")
XAI_RESULTS_PATH = os.path.join(PROJECT_ROOT, "xai_results.csv")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

PERSONA_TO_SIMPLICITY = {1: 20, 2: 40, 3: 60, 4: 78, 5: 92}
TABULAR_SIMPLICITY_DEFAULTS = {
    "SHAP": 72, "LIME": 87, "PFI": 91, "PDP": 85,
    "Permutation Feature Importance": 91, "Partial Dependence Plots": 85,
}

# XAI methods per type (must match update_excel_and_csv)
TABULAR_XAI = ["SHAP", "LIME", "PFI", "PDP"]
IMAGE_XAI = ["GradCAM", "GradCAM_pp", "Integrated_Gradients", "SmoothGrad", "LIME_Image", "SHAP_Image",
             "Saliency_Maps", "Guided_Backprop", "Guided_GradCAM", "ScoreCAM", "LayerCAM", "Attention_Maps"]
TEXT_XAI = ["LIME_Text", "SHAP_Text", "Integrated_Gradients_Text", "Attention_Weights", "Gradient_Text",
            "InputXGradient_Text", "DeepLIFT_Text", "Occlusion_Text", "Shapley_Sampling_Text", "Rationale_Extraction"]
TS_XAI = ["SHAP_TS", "LIME_TS", "Attention_TS", "Feature_Importance_TS", "Temporal_Attribution",
          "Integrated_Gradients_TS", "Gradient_TS", "Permutation_Importance_TS"]

XAI_BASES = {
    "GradCAM": {"fid": (0.75, 0.88), "sim": (65, 85), "stab": (0.7, 0.85), "loc": (0.7, 0.9)},
    "GradCAM_pp": {"fid": (0.78, 0.9), "sim": (60, 80), "stab": (0.72, 0.88), "loc": (0.72, 0.9)},
    "Integrated_Gradients": {"fid": (0.8, 0.92), "sim": (60, 75), "stab": (0.75, 0.88), "loc": (0.6, 0.8)},
    "SmoothGrad": {"fid": (0.7, 0.85), "sim": (55, 70), "stab": (0.65, 0.8), "loc": (0.5, 0.7)},
    "LIME_Image": {"fid": (0.65, 0.8), "sim": (75, 90), "stab": (0.5, 0.7), "loc": (0.5, 0.7)},
    "SHAP_Image": {"fid": (0.78, 0.92), "sim": (60, 75), "stab": (0.75, 0.9), "loc": (0.65, 0.82)},
    "Saliency_Maps": {"fid": (0.6, 0.75), "sim": (70, 85), "stab": (0.55, 0.72), "loc": (0.45, 0.65)},
    "Guided_Backprop": {"fid": (0.55, 0.7), "sim": (65, 80), "stab": (0.5, 0.65), "loc": (0.4, 0.6)},
    "Guided_GradCAM": {"fid": (0.72, 0.86), "sim": (68, 82), "stab": (0.68, 0.82), "loc": (0.68, 0.85)},
    "ScoreCAM": {"fid": (0.7, 0.84), "sim": (65, 80), "stab": (0.65, 0.8), "loc": (0.6, 0.78)},
    "LayerCAM": {"fid": (0.72, 0.86), "sim": (62, 78), "stab": (0.67, 0.82), "loc": (0.65, 0.82)},
    "Attention_Maps": {"fid": (0.68, 0.82), "sim": (70, 88), "stab": (0.65, 0.8), "loc": (0.6, 0.78)},
    "LIME_Text": {"fid": (0.7, 0.85), "sim": (78, 92), "stab": (0.55, 0.72), "loc": (0.5, 0.7)},
    "SHAP_Text": {"fid": (0.78, 0.9), "sim": (65, 80), "stab": (0.75, 0.88), "loc": (0.6, 0.78)},
    "Integrated_Gradients_Text": {"fid": (0.82, 0.94), "sim": (58, 72), "stab": (0.8, 0.92), "loc": (0.65, 0.8)},
    "Attention_Weights": {"fid": (0.65, 0.8), "sim": (75, 90), "stab": (0.6, 0.75), "loc": (0.55, 0.72)},
    "Gradient_Text": {"fid": (0.6, 0.75), "sim": (62, 78), "stab": (0.55, 0.7), "loc": (0.45, 0.65)},
    "InputXGradient_Text": {"fid": (0.68, 0.82), "sim": (60, 75), "stab": (0.65, 0.78), "loc": (0.5, 0.68)},
    "DeepLIFT_Text": {"fid": (0.72, 0.86), "sim": (58, 72), "stab": (0.7, 0.84), "loc": (0.55, 0.72)},
    "Occlusion_Text": {"fid": (0.65, 0.78), "sim": (72, 88), "stab": (0.6, 0.75), "loc": (0.5, 0.68)},
    "Shapley_Sampling_Text": {"fid": (0.75, 0.88), "sim": (60, 75), "stab": (0.7, 0.85), "loc": (0.58, 0.75)},
    "Rationale_Extraction": {"fid": (0.7, 0.85), "sim": (80, 95), "stab": (0.65, 0.8), "loc": (0.62, 0.8)},
    "SHAP_TS": {"fid": (0.75, 0.88), "sim": (65, 80), "stab": (0.72, 0.88), "loc": (0.58, 0.75)},
    "LIME_TS": {"fid": (0.68, 0.82), "sim": (72, 88), "stab": (0.58, 0.72), "loc": (0.5, 0.68)},
    "Attention_TS": {"fid": (0.7, 0.84), "sim": (70, 85), "stab": (0.65, 0.8), "loc": (0.55, 0.72)},
    "Feature_Importance_TS": {"fid": (0.72, 0.86), "sim": (75, 90), "stab": (0.7, 0.85), "loc": (0.5, 0.65)},
    "Temporal_Attribution": {"fid": (0.78, 0.9), "sim": (62, 78), "stab": (0.75, 0.88), "loc": (0.72, 0.88)},
    "Integrated_Gradients_TS": {"fid": (0.8, 0.92), "sim": (58, 72), "stab": (0.78, 0.9), "loc": (0.6, 0.78)},
    "Gradient_TS": {"fid": (0.62, 0.76), "sim": (60, 75), "stab": (0.6, 0.75), "loc": (0.45, 0.62)},
    "Permutation_Importance_TS": {"fid": (0.7, 0.84), "sim": (78, 92), "stab": (0.65, 0.8), "loc": (0.48, 0.65)},
    "SHAP": {"fid": (0.82, 0.94), "sim": (65, 80), "stab": (0.8, 0.92), "loc": None},
    "LIME": {"fid": (0.7, 0.85), "sim": (80, 95), "stab": (0.55, 0.72), "loc": None},
    "PFI": {"fid": (0.75, 0.88), "sim": (85, 98), "stab": (0.7, 0.85), "loc": None},
    "PDP": {"fid": (0.72, 0.86), "sim": (78, 92), "stab": (0.72, 0.88), "loc": None},
}


def _deterministic_frac(seed_str: str) -> float:
    """Returns 0-1 based on hash, reproducible."""
    h = hash(seed_str) % (2**32)
    return (h % 10000) / 10000.0


def persona_to_simplicity(interp: float, under: float, trust: float) -> float:
    """Map mean of 1-5 ratings to 0-100 Simplicity."""
    avg = (interp + under + trust) / 3.0
    low = int(avg) if avg < 5 else 4
    high = min(5, low + 1)
    frac = avg - low
    low_val = PERSONA_TO_SIMPLICITY.get(low, 60)
    high_val = PERSONA_TO_SIMPLICITY.get(high, 92)
    return low_val + frac * (high_val - low_val)


def fix_results_ai_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Merge Precision (space) into Precision, drop duplicate."""
    if "Precision " in df.columns and "Precision" in df.columns:
        prec_space = df["Precision "]
        prec_main = df["Precision"]
        df = df.drop(columns=["Precision ", "Precision"])
        df["Precision"] = prec_space.fillna(prec_main)
    elif "Precision " in df.columns:
        df = df.rename(columns={"Precision ": "Precision"})
    return df


def normalize_results_xai_simplicity(
    results_xai: pd.DataFrame,
    xai_results: pd.DataFrame,
    data_df: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize Simplicity to 0-100: persona-derived for tabular, deterministic for others."""
    rx = results_xai.copy()
    method_col = "XAI Method"
    ds_col = "Dataset ID"
    sim_col = "Simplicity"

    # Build dataset_id -> data_type
    id_to_type = {}
    if "dataset_id" in data_df.columns and "data_type" in data_df.columns:
        for _, r in data_df.iterrows():
            did = r["dataset_id"]
            dt = r.get("data_type")
            if pd.isna(dt) or not dt:
                dt = "tabular"
            id_to_type[str(did)] = str(dt).strip().lower()

    # Build persona ratings: dataset_id -> method -> (interp, under, trust)
    persona_ratings: Dict[str, Dict[str, tuple]] = {}
    for _, row in xai_results.iterrows():
        ds_id = str(row["dataset_id"])
        persona_ratings[ds_id] = {}
        for m in TABULAR_XAI:
            interp_col = f"interpretability_{m}"
            under_col = f"understanding_{m}"
            trust_col = f"trust_{m}"
            if interp_col in row and under_col in row and trust_col in row:
                i = row[interp_col]
                u = row[under_col]
                t = row[trust_col]
                if pd.notna(i) and pd.notna(u) and pd.notna(t):
                    persona_ratings[ds_id][m] = (float(i), float(u), float(t))

    def get_simplicity(row) -> float:
        ds_id = str(row[ds_col])
        method = str(row[method_col]).strip()
        if method == "Permutation Feature Importance":
            method = "PFI"
        elif method == "Partial Dependence Plots":
            method = "PDP"

        dt = id_to_type.get(ds_id, "tabular")
        current_sim = row.get(sim_col)

        if dt == "tabular":
            # Persona-derived or default
            if ds_id in persona_ratings and method in persona_ratings[ds_id]:
                i, u, t = persona_ratings[ds_id][method]
                return round(persona_to_simplicity(i, u, t), 2)
            return float(TABULAR_SIMPLICITY_DEFAULTS.get(method, 75))
        else:
            # Image/text/timeseries: deterministic estimate
            if current_sim is not None and not pd.isna(current_sim) and 0 <= current_sim <= 100:
                return float(current_sim)
            domain = "general"
            b = XAI_BASES.get(method, {"sim": (65, 80)})
            frac = _deterministic_frac(f"{dt}_{method}_{domain}_{ds_id}")
            sim = b["sim"][0] + frac * (b["sim"][1] - b["sim"][0])
            if domain in ("healthcare", "finance"):
                sim = min(100, sim * 1.05)
            return round(sim, 2)

    rx[sim_col] = rx.apply(get_simplicity, axis=1)
    return rx


def _estimate_xai_metrics_deterministic(
    data_type: str, method: str, domain: str, dataset_id: str
) -> Dict[str, Optional[float]]:
    """Deterministic estimate for Fidelity, Simplicity, Stability, Localization."""
    b = XAI_BASES.get(method, {"fid": (0.65, 0.8), "sim": (65, 80), "stab": (0.6, 0.75), "loc": (0.5, 0.7)})
    seed = f"{data_type}_{method}_{domain}_{dataset_id}"
    frac = _deterministic_frac(seed)

    def r(lo, hi):
        return lo + frac * (hi - lo)

    fid = r(b["fid"][0], b["fid"][1])
    sim = r(b["sim"][0], b["sim"][1])
    stab = r(b["stab"][0], b["stab"][1])
    loc = r(b["loc"][0], b["loc"][1]) if b.get("loc") else None

    if domain in ("healthcare", "finance"):
        sim = min(100, sim * 1.05)
        stab = min(1.0, stab * 1.02)

    return {"fidelity": fid, "simplicity": sim, "stability": stab, "localization": loc}


def main() -> int:
    if not os.path.exists(EXCEL_PATH):
        print(f"Excel not found: {EXCEL_PATH}")
        return 1
    if not os.path.exists(XAI_RESULTS_PATH):
        print(f"xai_results.csv not found: {XAI_RESULTS_PATH}")
        return 1

    print("Loading Excel and xai_results.csv...")
    xl = pd.ExcelFile(EXCEL_PATH)
    data_df = pd.read_excel(xl, sheet_name="data")
    results_ai_df = pd.read_excel(xl, sheet_name="results_ai")
    results_xai_df = pd.read_excel(xl, sheet_name="results_xai")
    ai_models_df = pd.read_excel(xl, sheet_name="ai models")
    xai_models_df = pd.read_excel(xl, sheet_name="xai methods")
    xai_results_df = pd.read_csv(XAI_RESULTS_PATH)

    # Fix results_ai
    results_ai_df = fix_results_ai_columns(results_ai_df)
    print("Fixed results_ai (Precision column)")

    # Normalize results_xai Simplicity
    results_xai_df = normalize_results_xai_simplicity(results_xai_df, xai_results_df, data_df)
    print("Normalized results_xai Simplicity to 0-100")

    # Ensure data_type in data_df
    if "data_type" not in data_df.columns:
        data_df["data_type"] = "tabular"
    data_df["data_type"] = data_df["data_type"].fillna("tabular").astype(str).str.strip().str.lower()

    # Build dataset_id -> data_type
    id_to_type = {}
    for _, r in data_df.iterrows():
        did = r["dataset_id"]
        dt = r.get("data_type", "tabular")
        id_to_type[str(did)] = str(dt).strip().lower() if dt else "tabular"

    # Infer type for IDs not in data (e.g. img_001, text_001)
    for ds_id in set(results_ai_df["dataset_id"].astype(str).tolist()) | set(
        results_xai_df["Dataset ID"].astype(str).tolist()
    ):
        if ds_id not in id_to_type:
            if ds_id.startswith("img_"):
                id_to_type[ds_id] = "image"
            elif ds_id.startswith("text_"):
                id_to_type[ds_id] = "text"
            elif ds_id.startswith("ts_"):
                id_to_type[ds_id] = "timeseries"
            else:
                id_to_type[ds_id] = "tabular"

    # Create data directories
    for dtype in ["tabular", "image", "text", "timeseries", "shared"]:
        d = os.path.join(DATA_ROOT, dtype)
        os.makedirs(d, exist_ok=True)

    # Export shared
    ai_models_df.to_csv(os.path.join(DATA_ROOT, "shared", "ai_model_definitions.csv"), index=False)
    xai_models_df.to_csv(os.path.join(DATA_ROOT, "shared", "xai_method_definitions.csv"), index=False)
    print("Exported shared/ai_model_definitions.csv, shared/xai_method_definitions.csv")

    # Copy survey docs if they exist; create default JSON fallbacks when DOCX absent
    survey_dir = os.path.join(DATA_ROOT, "shared", "survey")
    os.makedirs(survey_dir, exist_ok=True)
    docx_copied = 0
    for fname in ["JSON_1.docx", "JSON_2.docx", "JSON_3.docx"]:
        src = os.path.join(PROJECT_ROOT, fname)
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, os.path.join(survey_dir, fname))
            docx_copied += 1
    # If any DOCX files are missing, create default JSON survey files
    if docx_copied < 3:
        try:
            create_survey_script = os.path.join(
                os.path.dirname(__file__), "create_default_survey.py"
            )
            if os.path.exists(create_survey_script):
                import subprocess
                subprocess.run([sys.executable, create_survey_script], check=False, cwd=PROJECT_ROOT)
        except Exception as e:
            print(f"Note: Could not create default survey JSONs: {e}")

    # Split by data_type
    for dtype in ["tabular", "image", "text", "timeseries"]:
        type_dir = os.path.join(DATA_ROOT, dtype)
        ids_of_type = [k for k, v in id_to_type.items() if v == dtype]

        # metadata: from data_df, plus minimal rows for IDs only in results
        meta = data_df[data_df["dataset_id"].astype(str).isin(ids_of_type)].copy()
        missing_ids = [i for i in ids_of_type if i not in meta["dataset_id"].astype(str).tolist()]
        if missing_ids:
            extra = pd.DataFrame([
                {"dataset_id": did, "dataset_name": f"Dataset {did}", "data_type": dtype, "domain": "general"}
                for did in missing_ids
            ])
            meta = pd.concat([meta, extra], ignore_index=True)
        meta.to_csv(os.path.join(type_dir, "dataset_metadata.csv"), index=False)

        # results_ai
        rai = results_ai_df[results_ai_df["dataset_id"].astype(str).isin(ids_of_type)]
        rai.to_csv(os.path.join(type_dir, "ai_model_performance.csv"), index=False)

        # results_xai
        rxi = results_xai_df[results_xai_df["Dataset ID"].astype(str).isin(ids_of_type)]
        rxi.to_csv(os.path.join(type_dir, "xai_quantitative_metrics.csv"), index=False)

        # xai_results (qualitative)
        xai = xai_results_df[xai_results_df["dataset_id"].astype(str).isin(ids_of_type)]
        xai.to_csv(os.path.join(type_dir, "xai_qualitative_ratings.csv"), index=False)

        print(f"  {dtype}: {len(ids_of_type)} datasets, metadata={len(meta)}, results_ai={len(rai)}, results_xai={len(rxi)}, xai_results={len(xai)}")

    print(f"\nDone. Data written to {DATA_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
