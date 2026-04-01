from typing import Optional
from fastapi import FastAPI, Form, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import json
import os
import base64
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import matplotlib.pyplot as plt
from docx import Document
import openai
from openai import OpenAI
import re
import traceback
from werkzeug.utils import secure_filename
import uuid
import tempfile

# Import functions from the original app.py
from app import (
    LLMXAIAssistant,
    load_data_from_folders,
    get_available_datasets,
    estimate_xai_score_for_dataset,
    estimate_xai_score_for_new_dataset,
    extract_features_from_new_dataset,
    parse_uploaded_file,
    get_method_description,
    get_model_description,
    generate_rule_based_response
)

app = FastAPI(
    title="XAI Scoring Framework API",
    description="A comprehensive API for evaluating and scoring Explainable AI methods",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global repository storage: {data_type: repository_dict}
_global_repository = None
DATA_ROOT = "data"

def _sanitize_for_json(obj):
    """Replace NaN/inf with None so JSON responses remain valid."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if hasattr(obj, "__float__"):
        try:
            f = float(obj)
            return None if (np.isnan(f) or np.isinf(f)) else f
        except (ValueError, TypeError):
            return None
    return obj

@app.get("/")
async def root():
    return {
        "message": "XAI Scoring Framework API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "load_data": "/load-data",
            "datasets": "/datasets?data_type=tabular",
            "score_dataset": "/score-dataset",
            "score_features": "/score-features",
            "methods": "/api/methods",
            "domains": "/api/domains"
        }
    }

@app.get("/health")
async def health_check():
    global _global_repository
    return {
        "status": "healthy",
        "repository_loaded": _global_repository is not None and len(_global_repository) > 0
    }

@app.post("/load-data")
async def load_data_endpoint():
    """Load benchmark data from data/ folders."""
    global _global_repository
    try:
        repos = load_data_from_folders(DATA_ROOT)
        if repos:
            _global_repository = repos
            return {
                "success": True,
                "message": "Benchmark data loaded successfully!",
                "repository": "available"
            }
        else:
            raise HTTPException(status_code=500, detail="No data found. Run revise_and_split_data.py first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/datasets")
async def get_datasets(data_type: str = "tabular"):
    """Get available datasets for a data type."""
    datasets = get_available_datasets(data_type, DATA_ROOT)
    return datasets

@app.post("/score-dataset")
async def score_dataset(
    dataset_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    data_type: str = Form("tabular"),
    domain: str = Form("general"),
    top_k: int = Form(3),
    fidelity_weight: float = Form(0.25),
    stability_weight: float = Form(0.25),
    user_rating_weight: float = Form(0.25),
    simplicity_weight: float = Form(0.25),
    use_relevance_weighting: bool = Form(True),
    num_classes: Optional[int] = Form(None),
    image_width: Optional[int] = Form(None),
    image_height: Optional[int] = Form(None),
    channels: Optional[int] = Form(None),
    series_length: Optional[int] = Form(None),
    num_channels: Optional[int] = Form(None),
):
    global _global_repository
    try:
        if _global_repository is None or len(_global_repository) == 0:
            raise HTTPException(status_code=400, detail="No repository loaded. Please load data first.")

        if not dataset_id and (not file or not file.filename):
            raise HTTPException(
                status_code=400,
                detail="Either dataset_id or file must be provided."
            )

        repository = _global_repository.get(data_type)
        if not repository:
            raise HTTPException(status_code=400, detail=f"No data for type '{data_type}'")

        weights = {
            'fid': fidelity_weight,
            'stab': stability_weight,
            'rate': user_rating_weight,
            'simp': simplicity_weight
        }

        if file and file.filename:
            # Upload flow: parse file, extract features, score
            metadata = {}
            for key, val in [
                ("num_classes", num_classes),
                ("image_width", image_width),
                ("image_height", image_height),
                ("channels", channels),
                ("series_length", series_length),
                ("num_channels", num_channels),
            ]:
                if val is not None:
                    metadata[key] = int(val)
            content = await file.read()
            file_obj = BytesIO(content)
            df, meta = parse_uploaded_file(file_obj, data_type, metadata, filename=file.filename)
            features_extracted = extract_features_from_new_dataset(df, data_type=data_type, metadata=meta)
            results_tuple = estimate_xai_score_for_new_dataset(
                new_df=df, repository=repository, domain=domain, top_k=top_k,
                weights=weights, data_type=data_type, metadata=meta,
                use_relevance_weighting=use_relevance_weighting
            )
            features_dict = {
                'dataset_name': file.filename,
                'domain': domain,
                'data_type': data_type,
                **{k: v for k, v in features_extracted.items() if k != 'data_type'}
            }
        else:
            # Benchmark dataset flow
            results_tuple = estimate_xai_score_for_dataset(
                dataset_id, repository, domain, top_k, weights, data_type,
                use_relevance_weighting=use_relevance_weighting
            )
            entry = repository.get(dataset_id) or repository.get(str(dataset_id))
            if not entry and str(dataset_id).replace("-", "").isdigit():
                try:
                    entry = repository.get(int(dataset_id))
                except (ValueError, TypeError):
                    pass
            features_dict = {
                'dataset_name': entry.get('dataset_name', str(dataset_id)) if entry else str(dataset_id),
                'domain': entry.get('domain', '') if entry else '',
                'data_type': data_type
            }
            if entry:
                for k in ('size', 'feature_count', 'numeric_features', 'cat_features',
                         'image_width', 'image_height', 'channels', 'num_classes',
                         'avg_doc_length', 'vocab_size', 'max_length',
                         'series_length', 'num_channels'):
                    if entry.get(k) is not None:
                        features_dict[k] = entry[k]

        results = results_tuple[0]
        recommended_method = results_tuple[1]
        recommended_ai = results_tuple[2]
        similar_datasets = results_tuple[3]

        return _sanitize_for_json({
            "success": True,
            "results": results,
            "features": features_dict,
            "recommended_method": recommended_method,
            "recommended_ai": recommended_ai,
            "similar_datasets": similar_datasets
        })

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score-features")
async def score_features(
    num_features: int = Form(...),
    num_samples: int = Form(...),
    domain: str = Form("general"),
    top_k: int = Form(3),
    fidelity_weight: float = Form(0.25),
    stability_weight: float = Form(0.25),
    user_rating_weight: float = Form(0.25),
    simplicity_weight: float = Form(0.25),
    use_relevance_weighting: bool = Form(True)
):
    global _global_repository
    try:
        if _global_repository is None or len(_global_repository) == 0:
            raise HTTPException(status_code=400, detail="No repository loaded. Please load data first.")

        repository = _global_repository.get("tabular")
        if not repository:
            raise HTTPException(status_code=400, detail="No tabular data loaded.")

        weights = {
            'fid': fidelity_weight,
            'stab': stability_weight,
            'rate': user_rating_weight,
            'simp': simplicity_weight
        }

        np.random.seed(42)
        dummy_data = np.random.randn(num_samples, num_features)
        df = pd.DataFrame(dummy_data, columns=[f'feature_{i}' for i in range(num_features)])

        features = extract_features_from_new_dataset(df, data_type="tabular")

        results_tuple = estimate_xai_score_for_new_dataset(
            df, repository, domain, top_k, weights, data_type="tabular",
            use_relevance_weighting=use_relevance_weighting
        )
        
        # Extract the results dictionary from the tuple
        if isinstance(results_tuple, tuple) and len(results_tuple) >= 1:
            results = results_tuple[0]
            recommended_method = results_tuple[1] if len(results_tuple) > 1 else None
            recommended_ai = results_tuple[2] if len(results_tuple) > 2 else None
            similar_datasets = results_tuple[3] if len(results_tuple) > 3 else None
        else:
            results = results_tuple
            recommended_method = None
            recommended_ai = None
            similar_datasets = None
        
        return _sanitize_for_json({
            "success": True,
            "results": results,
            "features": features,
            "recommended_method": recommended_method,
            "recommended_ai": recommended_ai,
            "similar_datasets": similar_datasets
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/methods")
async def get_methods():
    """Get available XAI methods with detailed descriptions."""
    methods = [
        {
            "id": "SHAP",
            "name": "SHAP (SHapley Additive exPlanations)",
            "description": "Game theory-based approach to explain model predictions by calculating the contribution of each feature to the prediction.",
            "type": "Global and Local",
            "complexity": "High",
            "interpretability": "High"
        },
        {
            "id": "LIME",
            "name": "LIME (Local Interpretable Model-agnostic Explanations)",
            "description": "Creates local approximations of complex models to explain individual predictions.",
            "type": "Local",
            "complexity": "Medium",
            "interpretability": "High"
        },
        {
            "id": "PFI",
            "name": "Permutation Feature Importance",
            "description": "Measures feature importance by randomly permuting feature values and observing the impact on model performance.",
            "type": "Global",
            "complexity": "Low",
            "interpretability": "Medium"
        },
        {
            "id": "PDP",
            "name": "Partial Dependence Plots",
            "description": "Shows the relationship between a feature and the model's predictions while accounting for the average effect of other features.",
            "type": "Global",
            "complexity": "Low",
            "interpretability": "High"
        }
    ]
    return methods

@app.get("/api/domains")
async def get_domains():
    """Get available domains with descriptions."""
    domains = [
        {
            "id": "general",
            "name": "General",
            "description": "General purpose applications with standard interpretability requirements.",
            "bonus_factor": 1.0
        },
        {
            "id": "healthcare",
            "name": "Healthcare",
            "description": "Medical diagnosis and treatment applications requiring high interpretability and trust.",
            "bonus_factor": 1.2
        },
        {
            "id": "finance",
            "name": "Finance",
            "description": "Financial modeling and risk assessment with regulatory compliance requirements.",
            "bonus_factor": 1.15
        },
        {
            "id": "cybersecurity",
            "name": "Cybersecurity",
            "description": "Security and threat detection applications requiring explainable decisions.",
            "bonus_factor": 1.1
        },
        {
            "id": "autonomous_vehicles",
            "name": "Autonomous Vehicles",
            "description": "Self-driving and transportation systems requiring safety-critical explanations.",
            "bonus_factor": 1.25
        },
        {
            "id": "recommendation_systems",
            "name": "Recommendation Systems",
            "description": "Product and content recommendations requiring user trust and transparency.",
            "bonus_factor": 1.05
        }
    ]
    return domains

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 