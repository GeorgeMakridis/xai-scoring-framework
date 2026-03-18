# XAI Scoring Framework - Technical Report

## 1. System Overview

The XAI Scoring Framework is a comprehensive system for evaluating and scoring Explainable AI (XAI) methods. It provides both a web-based user interface and a REST API for external integration.

### 1.1 Architecture

- **Core Engine** (`app.py`) - XAI evaluation logic
- **Web Interface** (`web_app.py`) - Flask-based UI
- **REST API** (`api/main.py`) - FastAPI for external access

## 2. How to Use

### 2.1 Web Interface

1. **Start the application**:
   ```bash
   docker-compose up -d
   ```

2. **Access the web interface**: http://localhost:8501

3. **Load benchmark data** by clicking "Load Benchmark Data"

4. **Upload your dataset** and configure scoring parameters:
   - Select domain (Healthcare, Finance, etc.)
   - Adjust weight sliders for Fidelity, Stability, User Rating, Simplicity
   - Click "Score Dataset"

### 2.2 REST API

**Health Check**:
```bash
curl http://localhost:8000/health
```

**Load Data**:
```bash
curl -X POST http://localhost:8000/load-data
```

**Score Dataset**:
```bash
curl -X POST -F "dataset_file=@your_dataset.csv" \
  -F "domain=general" \
  -F "fidelity_weight=0.25" \
  -F "stability_weight=0.25" \
  -F "user_rating_weight=0.25" \
  -F "simplicity_weight=0.25" \
  http://localhost:8000/score-dataset
```

**Get Available Methods**:
```bash
curl http://localhost:8000/api/methods
```

## 3. Key Code Snippets

### 3.1 Core Scoring Algorithm

```python
def estimate_xai_score_for_new_dataset(new_df, repository, domain="general", top_k=3, weights=None):
    """Main scoring function"""
    if weights is None:
        weights = {"fid": 0.4, "stab": 0.3, "rate": 0.2, "simp": 0.1}
    
    # Extract features from new dataset
    new_features = extract_features_from_new_dataset(new_df)
    
    # Find similar datasets using cosine similarity
    similarities = []
    for ds_id, entry in repository.items():
        repo_vector = extract_repo_features(entry)
        sim = cosine_similarity(new_vector.reshape(1, -1), repo_vector.reshape(1, -1))[0][0]
        similarities.append((ds_id, sim, entry.get("dataset_name", f"Dataset {ds_id}")))
    
    # Get top-k similar datasets
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    top_ids = [ds_id for ds_id, _, _ in similarities[:top_k]]
    
    # Calculate final scores with domain bonus
    for method, sums in method_metrics.items():
        domain_bonus = get_domain_xai_bonus(domain, method, survey_jsons)
        overall = (weights["fid"] * avg_fid + 
                   weights["stab"] * avg_stab + 
                   weights["rate"] * avg_rate - 
                   weights["simp"] * (avg_simp / 100)) * domain_bonus
```

### 3.2 Web Interface Route

```python
@app.route('/score_dataset', methods=['POST'])
def score_dataset():
    """Score uploaded dataset"""
    global _global_repository
    
    # Get form data
    domain = request.form.get('domain', 'general')
    weights = {
        'fid': float(request.form.get('fidelity_weight', 0.25)),
        'stab': float(request.form.get('stability_weight', 0.25)),
        'rate': float(request.form.get('user_rating_weight', 0.25)),
        'simp': float(request.form.get('simplicity_weight', 0.25))
    }
    
    # Process dataset and return results
    results_tuple = estimate_xai_score_for_new_dataset(
        df, repository, domain, top_k, weights
    )
    
    return jsonify({
        'success': True,
        'results': results,
        'features': features,
        'recommended_method': recommended_method
    })
```

### 3.3 API Endpoint

```python
@app.post("/score-dataset")
async def score_dataset(
    dataset_file: UploadFile = File(...),
    domain: str = Form("general"),
    fidelity_weight: float = Form(0.25),
    stability_weight: float = Form(0.25),
    user_rating_weight: float = Form(0.25),
    simplicity_weight: float = Form(0.25)
):
    """Score uploaded dataset via API"""
    weights = {
        'fid': fidelity_weight,
        'stab': stability_weight,
        'rate': user_rating_weight,
        'simp': simplicity_weight
    }
    
    # Process dataset
    content = await dataset_file.read()
    df = pd.read_csv(BytesIO(content))
    features = extract_features_from_new_dataset(df)
    
    # Calculate scores
    results = estimate_xai_score_for_new_dataset(
        df, _global_repository, domain, top_k, weights
    )
    
    return {
        "success": True,
        "results": results,
        "features": features
    }
```

### 3.4 Feature Extraction

```python
def extract_features_from_new_dataset(df):
    """Extract features for similarity matching"""
    return {
        "feature_count": len(df.columns),
        "size": len(df),
        "numeric_features": len(df.select_dtypes(include=[np.number]).columns),
        "cat_features": len(df.select_dtypes(include=['object']).columns),
        "missing_ratio": df.isnull().sum().sum() / (len(df) * len(df.columns))
    }
```

## 4. Docker Configuration

### 4.1 Dockerfile

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y build-essential curl
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app
EXPOSE 8501 8000
CMD ["python3", "web_app.py"]
```

### 4.2 Docker Compose

```yaml
version: '3.8'
services:
  xai-web:
    image: xai-scoring-framework
    ports:
      - "8501:8501"
    command: ["python3", "web_app.py"]
    
  xai-api:
    image: xai-scoring-framework
    ports:
      - "8000:8000"
    command: ["python3", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 5. Scoring Metrics

The framework evaluates XAI methods using four key metrics:

- **Fidelity** (40% weight): How accurately XAI represents the model
- **Stability** (30% weight): Consistency of explanations
- **User Rating** (20% weight): User interpretability ratings
- **Simplicity** (10% weight): Ease of understanding (penalty)

Domain-specific bonuses are applied based on application requirements.

## 6. Supported XAI Methods

- **SHAP**: Game theory-based approach
- **LIME**: Local model approximations
- **PFI**: Permutation feature importance
- **PDP**: Partial dependence plots

## 7. Quick Start

```bash
# Clone repository
git clone https://github.com/GeorgeMakridis/xai-scoring-framework.git
cd xai-scoring-framework

# Build and run with Docker
docker-compose up -d

# Access web interface
open http://localhost:8501

# Test API
curl http://localhost:8000/health
```

The framework provides a complete solution for XAI method evaluation with both user-friendly web interface and programmatic API access. 