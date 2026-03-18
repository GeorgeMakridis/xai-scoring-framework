from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
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

app = Flask(__name__)
app.secret_key = 'xai-scoring-framework-secret-key-2024'
app.config['DEBUG'] = False
app.config['TESTING'] = False
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# Global repository storage: {data_type: repository_dict}
_global_repository = None
DATA_ROOT = "data"
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'zip', 'txt'}


def _sanitize_for_json(obj):
    """Replace NaN/inf with None so JSON is valid and parseable by browsers."""
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
    if hasattr(obj, '__float__'):  # numpy scalar
        try:
            f = float(obj)
            return None if (np.isnan(f) or np.isinf(f)) else f
        except (ValueError, TypeError):
            return None
    return obj

@app.route('/')
def index():
    resp = app.make_response(render_template('index.html'))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.route('/load-data')
def load_data_endpoint():
    """Load benchmark data from data/ folders."""
    global _global_repository
    try:
        print("Loading data from folders...")
        repos = load_data_from_folders(DATA_ROOT)
        if repos:
            _global_repository = repos
            session['data_loaded'] = True
            print(f"✅ Data loaded: {list(repos.keys())}")
            return jsonify({
                'success': True,
                'message': 'Benchmark data loaded successfully!',
                'repository': 'available'
            })
        else:
            print("❌ No data found in data/")
            return jsonify({
                'success': False,
                'error': 'No benchmark data found. Run scripts/data_management/revise_and_split_data.py first.'
            }), 500
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/data-status')
def data_status():
    """Check if benchmark data is loaded."""
    global _global_repository
    return jsonify({
        'data_loaded': _global_repository is not None and len(_global_repository) > 0,
        'repository_available': _global_repository is not None and len(_global_repository) > 0
    })

@app.route('/datasets')
def datasets_endpoint():
    """Get available datasets for a data type."""
    data_type = request.args.get('data_type', 'tabular')
    datasets = get_available_datasets(data_type, DATA_ROOT)
    return jsonify(datasets)

@app.route('/score_dataset', methods=['POST'])
def score_dataset():
    global _global_repository
    try:
        if _global_repository is None or len(_global_repository) == 0:
            return jsonify({'error': 'No repository loaded. Please load data first.'}), 400

        data = request.form
        dataset_id = data.get('dataset_id')
        file = request.files.get('file')
        domain = data.get('domain', 'general')
        data_type = data.get('data_type', 'tabular')
        top_k = int(data.get('top_k', 3))
        use_relevance_weighting = data.get('use_relevance_weighting', 'true').lower() in ('true', '1', 'yes')

        weights = {
            'fid': float(data.get('fidelity_weight', 0.25)),
            'stab': float(data.get('stability_weight', 0.25)),
            'rate': float(data.get('user_rating_weight', 0.25)),
            'simp': float(data.get('simplicity_weight', 0.25))
        }

        repository = _global_repository.get(data_type)
        if not repository:
            return jsonify({'error': f'No data for type "{data_type}"'}), 400

        if file and file.filename:
            # Upload flow: parse file, extract features, score
            metadata = {}
            for key in ('num_classes', 'image_width', 'image_height', 'channels',
                        'series_length', 'num_channels'):
                val = data.get(key)
                if val is not None and val != '':
                    try:
                        metadata[key] = int(float(val))
                    except (ValueError, TypeError):
                        pass
            df, meta = parse_uploaded_file(file, data_type, metadata)
            features = extract_features_from_new_dataset(df, data_type=data_type, metadata=meta)
            results_tuple = estimate_xai_score_for_new_dataset(
                new_df=df, repository=repository, domain=domain, top_k=top_k,
                weights=weights, data_type=data_type, metadata=meta,
                use_relevance_weighting=use_relevance_weighting
            )
            features_dict = {
                'dataset_name': file.filename,
                'domain': domain,
                'data_type': data_type,
                **{k: v for k, v in features.items() if k != 'data_type'}
            }
        else:
            # Pre-loaded dataset flow
            if not dataset_id:
                return jsonify({'error': 'Either upload a file or select a benchmark dataset.'}), 400
            results_tuple = estimate_xai_score_for_dataset(
                dataset_id, repository, domain, top_k, weights, data_type,
                use_relevance_weighting=use_relevance_weighting
            )
            entry = repository.get(dataset_id) or repository.get(str(dataset_id))
            if not entry and str(dataset_id).replace('-', '').isdigit():
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
                features_dict.update({k: entry.get(k) for k in (
                    'size', 'feature_count', 'numeric_features', 'cat_features',
                    'image_width', 'image_height', 'channels', 'num_classes',
                    'avg_doc_length', 'vocab_size', 'max_length',
                    'series_length', 'num_channels'
                ) if entry.get(k) is not None})

        results = results_tuple[0]
        recommended_method = results_tuple[1]
        recommended_ai = results_tuple[2]
        similar_datasets = results_tuple[3]

        response_data = {
            'success': True,
            'results': results,
            'features': features_dict,
            'recommended_method': recommended_method,
            'recommended_ai': recommended_ai,
            'similar_datasets': similar_datasets
        }
        if not similar_datasets and not results:
            response_data['message'] = 'No similar benchmarks found for your dataset.'
        return jsonify(_sanitize_for_json(response_data))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    global _global_repository
    try:
        if _global_repository is None or len(_global_repository) == 0:
            return jsonify({'error': 'No repository loaded'}), 400

        # Merge all type repos for chat context
        repository = {}
        for repo in _global_repository.values():
            repository.update({k: v for k, v in repo.items() if k != "__survey_info__"})
        if _global_repository:
            first_repo = next(iter(_global_repository.values()))
            if "__survey_info__" in first_repo:
                repository["__survey_info__"] = first_repo["__survey_info__"]
        question = request.json.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Generate response using the original function
        response = generate_rule_based_response(question, repository)
        
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/methods')
def get_methods():
    """Get available XAI methods."""
    methods = [
        {'id': 'SHAP', 'name': 'SHAP (SHapley Additive exPlanations)', 'description': 'Game theory-based approach to explain model predictions'},
        {'id': 'LIME', 'name': 'LIME (Local Interpretable Model-agnostic Explanations)', 'description': 'Local approximation of complex models'},
        {'id': 'PFI', 'name': 'Permutation Feature Importance', 'description': 'Measures feature importance by permutation'},
        {'id': 'PDP', 'name': 'Partial Dependence Plots', 'description': 'Shows relationship between features and predictions'}
    ]
    return jsonify(methods)

@app.route('/api/domains')
def get_domains():
    """Get available domains."""
    domains = [
        {'id': 'general', 'name': 'General', 'description': 'General purpose applications'},
        {'id': 'healthcare', 'name': 'Healthcare', 'description': 'Medical diagnosis and treatment'},
        {'id': 'finance', 'name': 'Finance', 'description': 'Financial modeling and risk assessment'},
        {'id': 'cybersecurity', 'name': 'Cybersecurity', 'description': 'Security and threat detection'},
        {'id': 'autonomous_vehicles', 'name': 'Autonomous Vehicles', 'description': 'Self-driving and transportation'},
        {'id': 'recommendation_systems', 'name': 'Recommendation Systems', 'description': 'Product and content recommendations'}
    ]
    return jsonify(domains)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'XAI Scoring Framework Web UI'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting XAI Scoring Framework Web UI at http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False) 