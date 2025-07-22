from fastapi import FastAPI, File, UploadFile, Form, HTTPException
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
    load_excel_data,
    load_json_from_docx,
    load_qualitative_ratings,
    build_repository,
    extract_features_from_new_dataset,
    extract_repo_features,
    get_domain_xai_bonus,
    estimate_xai_score_for_new_dataset,
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

# Global repository storage
_global_repository = None

def load_hardcoded_data():
    """Load the benchmark data files that are already in the project directory."""
    try:
        # Load Excel data
        excel_file_path = "Fame XAI scoring Framework_v2-2.xlsx"
        if os.path.exists(excel_file_path):
            with open(excel_file_path, 'rb') as f:
                excel_content = f.read()
                excel_data = load_excel_data(excel_content)
                if excel_data is None:
                    print(f"❌ Failed to load Excel data from {excel_file_path}")
                    return None
        else:
            print(f"❌ {excel_file_path} not found")
            return None
        
        # Load ratings
        ratings_file_path = "xai_results.csv"
        if os.path.exists(ratings_file_path):
            with open(ratings_file_path, 'rb') as f:
                ratings_content = f.read()
                ratings_data = load_qualitative_ratings(ratings_content)
                if ratings_data is None:
                    print(f"❌ Failed to load ratings data from {ratings_file_path}")
                    return None
        else:
            print(f"❌ {ratings_file_path} not found")
            return None
        
        # Load JSON files if they exist
        survey_jsons = []
        json_files = ["JSON_1.docx", "JSON_2.docx", "JSON_3.docx"]
        
        for json_file in json_files:
            if os.path.exists(json_file):
                with open(json_file, 'rb') as f:
                    json_content = f.read()
                    json_data = load_json_from_docx(json_content)
                    if json_data:
                        survey_jsons.append(json_data)
                        print(f"✅ Loaded {json_file}")
            else:
                print(f"⚠️  {json_file} not found")
        
        # Unpack the tuple returned by load_excel_data
        results_ai_df, results_xai_df, data_df, ai_models_df, xai_models_df = excel_data
        
        # Build repository
        repository = build_repository(
            data_df,
            results_ai_df,
            results_xai_df,
            ai_models_df,
            xai_models_df,
            survey_jsons if survey_jsons else None
        )
        
        print("✅ Benchmark data loaded successfully!")
        return repository
        
    except Exception as e:
        print(f"❌ Error loading hardcoded data: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.get("/")
async def root():
    return {
        "message": "XAI Scoring Framework API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "load_data": "/load-data",
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
        "repository_loaded": _global_repository is not None
    }

@app.post("/load-data")
async def load_data():
    """Load the hardcoded benchmark data."""
    global _global_repository
    try:
        repository = load_hardcoded_data()
        if repository:
            _global_repository = repository
            return {
                "success": True,
                "message": "Benchmark data loaded successfully!",
                "repository": "available"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to load benchmark data")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score-dataset")
async def score_dataset(
    dataset_file: UploadFile = File(...),
    domain: str = Form("general"),
    top_k: int = Form(3),
    fidelity_weight: float = Form(0.25),
    stability_weight: float = Form(0.25),
    user_rating_weight: float = Form(0.25),
    simplicity_weight: float = Form(0.25)
):
    global _global_repository
    try:
        if _global_repository is None:
            raise HTTPException(status_code=400, detail="No repository loaded. Please load data first.")
        
        repository = _global_repository
        
        # Get weights
        weights = {
            'fid': fidelity_weight,
            'stab': stability_weight,
            'rate': user_rating_weight,
            'simp': simplicity_weight
        }
        
        # Read and process the dataset
        content = await dataset_file.read()
        if dataset_file.filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(content))
        else:
            df = pd.read_excel(BytesIO(content))
        
        features = extract_features_from_new_dataset(df)
        
        # Estimate XAI scores
        results_tuple = estimate_xai_score_for_new_dataset(
            df, repository, domain, top_k, weights
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
        
        return {
            "success": True,
            "results": results,
            "features": features,
            "recommended_method": recommended_method,
            "recommended_ai": recommended_ai,
            "similar_datasets": similar_datasets
        }
        
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
    simplicity_weight: float = Form(0.25)
):
    global _global_repository
    try:
        if _global_repository is None:
            raise HTTPException(status_code=400, detail="No repository loaded. Please load data first.")
        
        repository = _global_repository
        
        # Get weights
        weights = {
            'fid': fidelity_weight,
            'stab': stability_weight,
            'rate': user_rating_weight,
            'simp': simplicity_weight
        }
        
        # Create a dummy dataset with the specified features
        np.random.seed(42)
        dummy_data = np.random.randn(num_samples, num_features)
        df = pd.DataFrame(dummy_data, columns=[f'feature_{i}' for i in range(num_features)])
        
        features = extract_features_from_new_dataset(df)
        
        # Estimate XAI scores
        results_tuple = estimate_xai_score_for_new_dataset(
            df, repository, domain, top_k, weights
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
        
        return {
            "success": True,
            "results": results,
            "features": features,
            "recommended_method": recommended_method,
            "recommended_ai": recommended_ai,
            "similar_datasets": similar_datasets
        }
        
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