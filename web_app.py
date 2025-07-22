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

app = Flask(__name__)
app.secret_key = 'xai-scoring-framework-secret-key-2024'
app.config['DEBUG'] = False
app.config['TESTING'] = False

# Global variables
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global repository storage (not in session)
_global_repository = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load-data')
def load_data_endpoint():
    """Load the hardcoded benchmark data."""
    global _global_repository
    try:
        print("Loading hardcoded data...")
        repository = load_hardcoded_data()
        if repository:
            _global_repository = repository
            session['data_loaded'] = True
            print(f"✅ Data loaded successfully! Repository: {_global_repository is not None}")
            return jsonify({
                'success': True,
                'message': 'Benchmark data loaded successfully!',
                'repository': 'available'
            })
        else:
            print("❌ Failed to load data")
            return jsonify({
                'success': False,
                'error': 'Failed to load benchmark data'
            }), 500
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/data-status')
def data_status():
    """Check if benchmark data is loaded."""
    global _global_repository
    return jsonify({
        'data_loaded': _global_repository is not None,
        'repository_available': _global_repository is not None
    })

@app.route('/score_dataset', methods=['POST'])
def score_dataset():
    global _global_repository
    try:
        if _global_repository is None:
            return jsonify({'error': 'No repository loaded. Please load data files first.'}), 400
        
        repository = _global_repository
        
        # Get form data
        domain = request.form.get('domain', 'general')
        top_k = int(request.form.get('top_k', 3))
        
        # Get weights
        weights = {
            'fid': float(request.form.get('fidelity_weight', 0.25)),
            'stab': float(request.form.get('stability_weight', 0.25)),
            'rate': float(request.form.get('user_rating_weight', 0.25)),
            'simp': float(request.form.get('simplicity_weight', 0.25))
        }
        
        # Handle file upload
        if 'dataset_file' in request.files:
            file = request.files['dataset_file']
            if file and file.filename != '':
                if allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    
                    # Load and process the dataset
                    df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)
                    features = extract_features_from_new_dataset(df)
                    
                    # Estimate XAI scores
                    print(f"🔍 Scoring dataset with {len(df)} rows, {len(df.columns)} columns")
                    print(f"🔍 Domain: {domain}, Top-k: {top_k}")
                    print(f"🔍 Weights: {weights}")
                    
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
                    
                    print(f"🔍 Results: {results}")
                    print(f"🔍 Recommended Method: {recommended_method}")
                    print(f"🔍 Recommended AI: {recommended_ai}")
                    print(f"🔍 Similar Datasets: {similar_datasets}")
                    
                    return jsonify({
                        'success': True,
                        'results': results,
                        'features': features,
                        'recommended_method': recommended_method,
                        'recommended_ai': recommended_ai,
                        'similar_datasets': similar_datasets
                    })
                else:
                    return jsonify({'error': 'Invalid file type'}), 400
            else:
                return jsonify({'error': 'No file uploaded'}), 400
        else:
            return jsonify({'error': 'No dataset file provided'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    global _global_repository
    try:
        if _global_repository is None:
            return jsonify({'error': 'No repository loaded'}), 400
        
        repository = _global_repository
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
    # Force production mode
    import os
    os.environ['FLASK_ENV'] = 'production'
    os.environ['FLASK_DEBUG'] = '0'
    
    # Use waitress for production
    from waitress import serve
    print("🚀 Starting XAI Scoring Framework Web UI with Waitress...")
    serve(app, host='0.0.0.0', port=8501, threads=4) 