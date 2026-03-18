import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import base64
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import matplotlib.pyplot as plt
from docx import Document
import openai
from openai import OpenAI
import re
import traceback

# Set page configuration
st.set_page_config(
    page_title="XAI Recommendation System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for repository persistence and recommendation results
if 'repository' not in st.session_state:
    st.session_state['repository'] = None
if 'recommendation_results' not in st.session_state:
    st.session_state['recommendation_results'] = None
if 'uploaded_dataset_features' not in st.session_state:
    st.session_state['uploaded_dataset_features'] = None
if 'selected_domain' not in st.session_state:
    st.session_state['selected_domain'] = "General"

# Add custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e89ae;
        color: white;
    }
    .recommendation-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-container {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .info-text {
        color: #666;
        font-size: 14px;
    }
    .header-text {
        font-weight: bold;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)


###############################################################################
# LLM Assistant Class
###############################################################################

class LLMXAIAssistant:
    """XAI Assistant that leverages an LLM API for generating responses."""

    def __init__(self, api_key=None):
        """Initialize the LLM XAI Assistant."""
        self.api_key = api_key
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
        self.model = "gpt-4"  # Default model, can be changed

    def get_repository_context(self, repository):
        """Extract relevant information from repository to provide as context to LLM."""
        context = {}

        # Extract survey information
        survey_jsons = repository.get("__survey_info__", {})

        # Get XAI method information
        xai_methods = {}
        if "json1" in survey_jsons:
            xai_methods = {
                "descriptions": survey_jsons["json1"].get("xai_method_descriptions", {}),
                "strengths": survey_jsons["json1"].get("xai_method_strengths", {}),
                "weaknesses": survey_jsons["json1"].get("xai_method_weaknesses", {})
            }

        # Get AI model information
        ai_models = {}
        if "json2" in survey_jsons:
            ai_models = survey_jsons["json2"].get("ai_model_descriptions", {})

        # Get domain information
        domains = {}
        if "json3" in survey_jsons:
            domains = survey_jsons["json3"].get("xai_usage_in_industries", {})

        # Add to context
        context["xai_methods"] = xai_methods
        context["ai_models"] = ai_models
        context["domains"] = domains

        # Get a sample of benchmark datasets for context
        datasets = []
        for ds_id, entry in repository.items():
            if not isinstance(ds_id, (int, str)) or ds_id in ("__survey_info__", "__domain_relevance__", "__dataset_relevance__"):
                continue
            if len(datasets) >= 5:  # Limit to 5 examples to keep context manageable
                break
            datasets.append({
                "id": ds_id,
                "name": entry.get("dataset_name", f"Dataset {ds_id}"),
                "domain": entry.get("domain", "unknown"),
                "size": entry.get("size", 0),
                "feature_count": entry.get("feature_count", 0)
            })

        context["benchmark_datasets"] = datasets

        # Add recommendation results if available
        if st.session_state.get('recommendation_results'):
            context["recommendation_results"] = st.session_state['recommendation_results']

        # Add uploaded dataset features if available
        if st.session_state.get('uploaded_dataset_features'):
            context["uploaded_dataset_features"] = st.session_state['uploaded_dataset_features']

        return context

    def generate_system_prompt(self, repository):
        """Generate system prompt with context from repository."""
        context = self.get_repository_context(repository)

        system_prompt = """You are an XAI Assistant specializing in explainable AI methods and models. 
Your purpose is to help users understand AI models, XAI techniques, and choose the right methods for their needs.

You have access to the following repository information about XAI methods, AI models, and domain-specific applications:
"""

        # Add XAI method information
        system_prompt += "\n## XAI Methods\n"
        for method, desc in context["xai_methods"].get("descriptions", {}).items():
            system_prompt += f"- {method}: {desc}\n"
            # Add strengths and weaknesses if available
            if method in context["xai_methods"].get("strengths", {}):
                strengths = context["xai_methods"]["strengths"][method]
                system_prompt += "  Strengths: " + ", ".join(strengths[:3]) + "\n"
            if method in context["xai_methods"].get("weaknesses", {}):
                weaknesses = context["xai_methods"]["weaknesses"][method]
                system_prompt += "  Weaknesses: " + ", ".join(weaknesses[:3]) + "\n"

        # Add AI model information
        system_prompt += "\n## AI Models\n"
        for model, desc in context["ai_models"].items():
            system_prompt += f"- {model}: {desc}\n"

        # Add domain information
        system_prompt += "\n## Domain Applications\n"
        for domain, info in context["domains"].items():
            system_prompt += f"- {domain.capitalize()}\n"
            if "preferred_methods" in info:
                system_prompt += f"  Preferred methods: {', '.join(info['preferred_methods'])}\n"
            if "challenges" in info:
                system_prompt += f"  Challenges: {', '.join(info['challenges'])}\n"

        # Add benchmark dataset examples
        system_prompt += "\n## Example Benchmark Datasets\n"
        for ds in context["benchmark_datasets"]:
            system_prompt += f"- {ds['name']} (Domain: {ds['domain']}, Size: {ds['size']}, Features: {ds['feature_count']})\n"

        # Add recommendation results if available
        if "recommendation_results" in context:
            results = context["recommendation_results"]
            system_prompt += "\n## Recent Recommendation Results\n"
            system_prompt += f"Dataset: {results['dataset_name']}\n"
            system_prompt += f"Domain: {results['domain']}\n"

            if "uploaded_dataset_features" in context:
                features = context["uploaded_dataset_features"]
                system_prompt += f"Features: {features['feature_count']} total, {features['numeric_features']} numeric, {features['cat_features']} categorical\n"
                system_prompt += f"Size: {features['size']} samples, {features['missing_ratio']:.2%} missing values\n"

            system_prompt += f"Recommended AI Model: {results['recommended_ai']}\n"
            system_prompt += f"Recommended XAI Method: {results['recommended_xai']}\n"

            system_prompt += "Method Scores:\n"
            for method, scores in results['estimated_scores'].items():
                system_prompt += f"- {method}: Fidelity: {scores['avg_fidelity']:.2f}, Stability: {scores['avg_stability']:.2f}, Rating: {scores['avg_rating']:.2f}, Overall: {scores['overall_score']:.2f}\n"

        # Add guidelines
        system_prompt += """
## Response Guidelines
- Provide informative responses about XAI methods, AI models, and domain applications
- If asked about specific methods or models, include their strengths and weaknesses
- When comparing methods, highlight key differences in approach and best use cases
- If the user asks about recommendations, reference the latest recommendation results if available
- If asked about "my dataset" or "my data", refer to the information from the uploaded dataset
- Format your responses with Markdown for better readability
- Keep answers focused on the XAI domain
- If you don't have specific information on a topic, provide general best practices rather than making up information

Remember that your purpose is to help users understand XAI and choose appropriate methods for their datasets and domains.
"""

        return system_prompt

    def generate_response(self, question, repository):
        """Generate response to user question using the LLM API."""
        if not self.api_key or not self.client:
            return "Please configure an API key for the LLM service in the sidebar."

        try:
            # Generate system prompt with repository context
            system_prompt = self.generate_system_prompt(repository)

            # Remove emojis and other special characters that might cause encoding issues
            # This regex pattern removes emoji and other non-ASCII characters
            emoji_pattern = re.compile("["
                                       u"\U0001F600-\U0001F64F"  # emoticons
                                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                       u"\U0001F700-\U0001F77F"  # alchemical symbols
                                       u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                                       u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                       u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                       u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                                       u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                       u"\U00002702-\U000027B0"  # Dingbats
                                       u"\U000024C2-\U0001F251"
                                       "]+", flags=re.UNICODE)

            # Clean the system prompt
            clean_system_prompt = emoji_pattern.sub(r'', system_prompt)

            # Get response from LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": clean_system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.7,
                max_tokens=800
            )

            # Extract and return response content
            return response.choices[0].message.content

        except Exception as e:
            error_details = traceback.format_exc()
            st.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}\n\nFalling back to rule-based response."


###############################################################################
# Helper Functions
###############################################################################

def load_excel_data(file_bytes):
    """Load Excel data from uploaded file bytes."""
    try:
        xl = pd.read_excel(BytesIO(file_bytes), sheet_name=None, engine='openpyxl')

        results_ai_df = xl.get("results_ai")
        results_xai_df = xl.get("results_xai")
        data_df = xl.get("data")
        ai_models_df = xl.get("ai models")
        xai_models_df = xl.get("xai methods")

        # Basic checks
        if results_ai_df is None:
            st.error("Sheet 'results_ai' not found in Excel file.")
            return None
        if results_xai_df is None:
            st.error("Sheet 'results_xai' not found in Excel file.")
            return None
        if data_df is None:
            st.error("Sheet 'data' not found in Excel file.")
            return None
        if ai_models_df is None:
            st.error("Sheet 'ai models' not found in Excel file.")
            return None
        if xai_models_df is None:
            st.error("Sheet 'xai methods' not found in Excel file.")
            return None

        return results_ai_df, results_xai_df, data_df, ai_models_df, xai_models_df

    except Exception as e:
        st.error(f"Error loading Excel data: {e}")
        return None


def load_json_from_docx(file_bytes):
    """Read JSON content from uploaded .docx file bytes."""
    try:
        doc = Document(BytesIO(file_bytes))
        full_text = "\n".join([para.text for para in doc.paragraphs])
        data = json.loads(full_text)
        return data
    except Exception as e:
        st.error(f"Error loading JSON from DOCX: {e}")
        return None


def parse_uploaded_file(file_storage, data_type, metadata=None, filename=None):
    """
    Parse uploaded file and return (df, metadata) for feature extraction.
    Supports: tabular (csv/xlsx), text (csv), timeseries (csv), image (zip or csv with metadata).
    file_storage: file-like object (Flask FileStorage, BytesIO, or FastAPI UploadFile content as BytesIO).
    filename: optional, for extension detection when file_storage has no .filename attribute.
    """
    import zipfile
    metadata = metadata or {}
    fn = filename or getattr(file_storage, "filename", None) or ""
    ext = (fn or "").rsplit(".", 1)[-1].lower() if fn else ""

    if data_type == "tabular":
        if ext == "csv":
            df = pd.read_csv(file_storage)
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(file_storage)
        else:
            raise ValueError("Tabular data must be CSV or Excel (.csv, .xlsx, .xls)")
        return df, metadata

    if data_type == "text":
        if ext != "csv":
            raise ValueError("Text data must be CSV with a text column")
        df = pd.read_csv(file_storage)
        return df, metadata

    if data_type == "timeseries":
        if ext != "csv":
            raise ValueError("Time series data must be CSV")
        df = pd.read_csv(file_storage)
        return df, metadata

    if data_type == "image":
        if ext == "zip":
            with zipfile.ZipFile(file_storage, "r") as zf:
                names = [n for n in zf.namelist() if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]
                size = len(names)
                img_w, img_h, ch = 224, 224, 3
                if names:
                    try:
                        from PIL import Image as PILImage
                        with zf.open(names[0]) as f:
                            img = PILImage.open(f)
                            img_w, img_h = img.size[0], img.size[1]
                            ch = len(img.getbands()) if img.mode else 3
                    except Exception:
                        pass
                metadata = dict(metadata)
                metadata.update(size=size, image_width=img_w, image_height=img_h, channels=ch)
                df = pd.DataFrame({"path": names}) if size else pd.DataFrame()
                return df, metadata
        elif ext == "csv":
            df = pd.read_csv(file_storage)
            return df, metadata
        raise ValueError("Image data must be a ZIP of images or CSV with image paths")

    raise ValueError(f"Unsupported data type: {data_type}")


def _parse_qualitative_ratings_df(df):
    """Parse xai_results DataFrame into grouped ratings. Returns None if required cols missing."""
    required_cols = [
        "dataset_id",
        "interpretability_SHAP", "interpretability_LIME", "interpretability_PFI", "interpretability_PDP",
        "understanding_SHAP", "understanding_LIME", "understanding_PFI", "understanding_PDP",
        "trust_SHAP", "trust_LIME", "trust_PFI", "trust_PDP"
    ]
    if not all(c in df.columns for c in required_cols):
        return None
    df = df.copy()
    df["SHAP"] = (df["interpretability_SHAP"].fillna(3) + df["understanding_SHAP"].fillna(3) + df["trust_SHAP"].fillna(3)) / 3.0
    df["LIME"] = (df["interpretability_LIME"].fillna(3) + df["understanding_LIME"].fillna(3) + df["trust_LIME"].fillna(3)) / 3.0
    df["PFI"] = (df["interpretability_PFI"].fillna(3) + df["understanding_PFI"].fillna(3) + df["trust_PFI"].fillna(3)) / 3.0
    df["PDP"] = (df["interpretability_PDP"].fillna(3) + df["understanding_PDP"].fillna(3) + df["trust_PDP"].fillna(3)) / 3.0
    rating_cols = ["SHAP", "LIME", "PFI", "PDP"]
    legacy_prefixes = {"interpretability_SHAP", "interpretability_LIME", "interpretability_PFI", "interpretability_PDP"}
    for col in df.columns:
        if col.startswith("interpretability_") and col not in legacy_prefixes:
            suffix = col.replace("interpretability_", "")
            ucol = f"understanding_{suffix}"
            tcol = f"trust_{suffix}"
            if ucol in df.columns and tcol in df.columns:
                df[suffix] = (df[col].fillna(3) + df[ucol].fillna(3) + df[tcol].fillna(3)) / 3.0
                rating_cols.append(suffix)
    return df.groupby("dataset_id")[rating_cols].mean().reset_index()


def load_qualitative_ratings(file_bytes):
    """Load CSV containing XAI method ratings.
    Supports both legacy (SHAP, LIME, PFI, PDP) and extended methods (GradCAM, LIME_Text, etc.).
    """
    try:
        df = pd.read_csv(BytesIO(file_bytes))
        result = _parse_qualitative_ratings_df(df)
        if result is None:
            st.error("Required columns not found in ratings CSV.")
            return None
        return result
    except Exception as e:
        st.error(f"Error loading ratings CSV: {e}")
        return None


def _load_relevance_maps(data_root: str):
    """Load domain_relevance and dataset_relevance from data/shared. Returns (domain_map, dataset_map)."""
    shared = os.path.join(data_root, "shared")
    domain_map = {}
    dataset_map = {}
    domain_path = os.path.join(shared, "domain_relevance.csv")
    if os.path.exists(domain_path):
        try:
            df = pd.read_csv(domain_path)
            for _, row in df.iterrows():
                k = (str(row.get("domain_from", "")).strip().lower(), str(row.get("domain_to", "")).strip().lower())
                if k[0] and k[1]:
                    domain_map[k] = float(row.get("relevance", 0.5))
        except Exception:
            pass
    ds_path = os.path.join(shared, "dataset_relevance.csv")
    if os.path.exists(ds_path):
        try:
            df = pd.read_csv(ds_path)
            if "dataset_id_a" in df.columns and "dataset_id_b" in df.columns:
                for _, row in df.iterrows():
                    a, b = str(row["dataset_id_a"]).strip(), str(row["dataset_id_b"]).strip()
                    if a and b:
                        rel = float(row.get("relevance", 0.5))
                        dataset_map[(a, b)] = rel
                        dataset_map[(b, a)] = rel
            elif "dataset_id" in df.columns and "domain" in df.columns:
                for _, row in df.iterrows():
                    k = (str(row["dataset_id"]).strip(), str(row["domain"]).strip().lower())
                    if k[0] and k[1]:
                        dataset_map[k] = float(row.get("relevance", 0.5))
        except Exception:
            pass
    return domain_map, dataset_map


def get_dataset_relevance(dataset_domain, query_domain, domain_relevance_map, dataset_relevance_map=None, ds_id_a=None, ds_id_b=None):
    """Return 0-1 relevance. Uses domain_relevance first; falls back to dataset_relevance if provided."""
    d_from = (dataset_domain or "").strip().lower()
    d_to = (query_domain or "").strip().lower()
    if not d_from:
        d_from = "general"
    if not d_to:
        d_to = "general"
    key = (d_from, d_to)
    rel = domain_relevance_map.get(key)
    if rel is not None:
        return rel
    if dataset_relevance_map and ds_id_a is not None and ds_id_b is not None:
        pair_key = (str(ds_id_a).strip(), str(ds_id_b).strip())
        rel = dataset_relevance_map.get(pair_key)
        if rel is not None:
            return rel
    return 0.5


def _build_tfidf_cache(repository):
    """Build TF-IDF matrix from dataset_name + domain + description. Attach to repository."""
    texts = []
    ids = []
    for ds_id, e in repository.items():
        if not isinstance(ds_id, (int, str)) or ds_id in ("__survey_info__", "__domain_relevance__", "__dataset_relevance__"):
            continue
        name = str(e.get("dataset_name", "") or "")
        domain = str(e.get("domain", "") or "")
        desc = str(e.get("description", "") or "")
        text = f"{name} {domain} {desc}".lower()[:5000]
        texts.append(text)
        ids.append(ds_id)
    if not texts:
        repository["__tfidf_cache__"] = None
        return
    try:
        vectorizer = TfidfVectorizer(max_features=500, stop_words="english", min_df=1)
        X = vectorizer.fit_transform(texts)
        id_to_idx = {str(i): idx for idx, i in enumerate(ids)}
        id_to_idx.update({i: idx for idx, i in enumerate(ids)})
        repository["__tfidf_cache__"] = {
            "vectorizer": vectorizer,
            "X": X,
            "id_to_idx": id_to_idx,
            "ids": ids,
        }
    except Exception:
        repository["__tfidf_cache__"] = None


def compute_description_similarity(query_id, candidate_id, repository, query_text=None):
    """
    Return 0-1 cosine similarity from TF-IDF vectors.
    query_text: for uploaded datasets, pass "name domain" string; else use query_id to look up.
    """
    cache = repository.get("__tfidf_cache__")
    if cache is None:
        return 0.5
    id_to_idx = cache["id_to_idx"]
    X = cache["X"]
    qid = str(query_id)
    cid = str(candidate_id)
    if query_text is not None:
        try:
            q_vec = cache["vectorizer"].transform([query_text.lower()[:5000]])
            c_idx = id_to_idx.get(cid)
            if c_idx is None:
                return 0.5
            c_vec = X[c_idx : c_idx + 1]
            sim = cosine_similarity(q_vec, c_vec)[0][0]
            return float(np.clip(sim, 0, 1))
        except Exception:
            return 0.5
    q_idx = id_to_idx.get(qid)
    c_idx = id_to_idx.get(cid)
    if q_idx is None or c_idx is None:
        return 0.5
    try:
        sim = cosine_similarity(X[q_idx : q_idx + 1], X[c_idx : c_idx + 1])[0][0]
        return float(np.clip(sim, 0, 1))
    except Exception:
        return 0.5


def load_data_from_folders(data_root: str = "data") -> dict:
    """Load repositories from data/{type}/ folders. Returns {data_type: repository_dict}."""
    result = {}
    shared_path = os.path.join(data_root, "shared")
    if not os.path.exists(shared_path):
        return result
    domain_relevance_map, dataset_relevance_map = _load_relevance_maps(data_root)
    ai_models_df = pd.read_csv(os.path.join(shared_path, "ai_model_definitions.csv"))
    xai_models_df = pd.read_csv(os.path.join(shared_path, "xai_method_definitions.csv"))
    survey_jsons = {}
    survey_dir = os.path.join(shared_path, "survey")
    if os.path.exists(survey_dir):
        docx_names = ["JSON_1.docx", "JSON_2.docx", "JSON_3.docx"]
        json_names = ["json1.json", "json2.json", "json3.json"]
        for i in range(3):
            key = f"json{i + 1}"
            docx_path = os.path.join(survey_dir, docx_names[i])
            json_path = os.path.join(survey_dir, json_names[i])
            j = None
            if os.path.exists(docx_path):
                with open(docx_path, "rb") as f:
                    j = load_json_from_docx(f.read())
            if j is None and os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    j = json.load(f)
            if j:
                survey_jsons[key] = j
    survey_jsons = survey_jsons if survey_jsons else None

    for dtype in ["tabular", "image", "text", "timeseries"]:
        type_dir = os.path.join(data_root, dtype)
        if not os.path.exists(type_dir):
            continue
        meta_path = os.path.join(type_dir, "dataset_metadata.csv")
        rai_path = os.path.join(type_dir, "ai_model_performance.csv")
        rxi_path = os.path.join(type_dir, "xai_quantitative_metrics.csv")
        xai_path = os.path.join(type_dir, "xai_qualitative_ratings.csv")
        if not all(os.path.exists(p) for p in [meta_path, rai_path, rxi_path, xai_path]):
            continue
        data_df = pd.read_csv(meta_path)
        results_ai_df = pd.read_csv(rai_path)
        results_xai_df = pd.read_csv(rxi_path)
        xai_results_df = pd.read_csv(xai_path)
        repo = build_repository(
            data_df, results_ai_df, results_xai_df,
            ai_models_df, xai_models_df, survey_jsons
        )
        ratings_df = _parse_qualitative_ratings_df(xai_results_df)
        if ratings_df is not None:
            for _, row in ratings_df.iterrows():
                ds_id = row["dataset_id"]
                if ds_id in repo:
                    ratings = {}
                    for col in ["SHAP", "LIME", "PFI", "PDP"]:
                        if col in row and pd.notna(row[col]):
                            ratings[col] = float(row[col])
                    for col in row.index:
                        if col not in ("dataset_id", "SHAP", "LIME", "PFI", "PDP") and pd.notna(row.get(col)):
                            try:
                                ratings[col] = float(row[col])
                            except (ValueError, TypeError):
                                pass
                    repo[ds_id]["xai_method_ratings"] = ratings
        repo["__domain_relevance__"] = domain_relevance_map
        repo["__dataset_relevance__"] = dataset_relevance_map
        _build_tfidf_cache(repo)
        result[dtype] = repo
    return result


def get_available_datasets(data_type: str, data_root: str = "data") -> list:
    """Return list of {dataset_id, dataset_name, domain, ...} for dropdown."""
    type_dir = os.path.join(data_root, data_type)
    meta_path = os.path.join(type_dir, "dataset_metadata.csv")
    if not os.path.exists(meta_path):
        return []
    df = pd.read_csv(meta_path)
    cols = ["dataset_id", "dataset_name", "domain"]
    available = [c for c in cols if c in df.columns]
    return df[available].fillna("").to_dict("records")


def estimate_xai_score_for_dataset(
    dataset_id, repository, domain="general", top_k=3, weights=None, data_type="tabular", use_relevance_weighting=True
):
    """Get recommendations for a pre-loaded dataset by ID. No file upload."""
    if weights is None:
        weights = {"fid": 0.4, "stab": 0.3, "rate": 0.2, "simp": 0.1}
    entry = repository.get(dataset_id) or repository.get(str(dataset_id))
    if not entry and str(dataset_id).replace("-", "").isdigit():
        try:
            entry = repository.get(int(dataset_id))
        except (ValueError, TypeError):
            pass
    if not entry:
        return {}, None, None, []
    new_vector = extract_repo_features(entry)
    new_vector = np.nan_to_num(new_vector, nan=0.0)
    target_type = str(data_type or "tabular").strip().lower()

    dataset_relevance_map = repository.get("__dataset_relevance__") or {}
    alpha = SIMILARITY_STRUCTURAL_WEIGHT
    similarities = []
    for ds_id, e in repository.items():
        if not isinstance(ds_id, (int, str)) or ds_id in ("__survey_info__", "__domain_relevance__", "__dataset_relevance__"):
            continue
        raw = e.get("data_type")
        entry_data_type = "tabular" if (raw is None or (hasattr(pd, "isna") and pd.isna(raw))) else str(raw).strip().lower()
        if not entry_data_type or entry_data_type == "nan":
            entry_data_type = "tabular"
        if entry_data_type != target_type:
            continue
        repo_vector = extract_repo_features(e)
        structural_sim = _compute_similarity(new_vector, repo_vector)
        semantic_sim = compute_description_similarity(dataset_id, ds_id, repository)
        pair_rel = dataset_relevance_map.get((str(dataset_id), str(ds_id))) or dataset_relevance_map.get((str(ds_id), str(dataset_id)))
        if pair_rel is not None:
            semantic_sim = 0.7 * semantic_sim + 0.3 * float(pair_rel)
        final_sim = alpha * structural_sim + (1 - alpha) * semantic_sim
        similarities.append((ds_id, final_sim, e.get("dataset_name", f"Dataset {ds_id}")))

    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    top_ids = [ds_id for ds_id, _, _ in similarities[:top_k]]
    similar_datasets = [(ds_id, sim, name) for ds_id, sim, name in similarities[:top_k]]

    if not similar_datasets:
        return {}, None, None, []

    method_metrics = {}
    count = {}
    survey_jsons = repository.get("__survey_info__", {})
    domain_relevance_map = repository.get("__domain_relevance__") or {}
    dataset_relevance_map = repository.get("__dataset_relevance__") or {}
    if not use_relevance_weighting:
        domain_relevance_map = {}
        dataset_relevance_map = {}

    for ds_id in top_ids:
        e = repository.get(ds_id, {})
        xai_results = e.get("xai_results", {})
        ratings = e.get("xai_method_ratings", {})
        dataset_domain = e.get("domain")
        if dataset_domain is None or (hasattr(pd, "isna") and pd.isna(dataset_domain)):
            dataset_domain = ""
        rel = get_dataset_relevance(
            dataset_domain, domain, domain_relevance_map, dataset_relevance_map,
            ds_id_a=dataset_id, ds_id_b=ds_id
        ) if use_relevance_weighting else 1.0

        for method, metrics in xai_results.items():
            if method not in method_metrics:
                method_metrics[method] = {"fidelity": 0, "simplicity": 0, "stability": 0, "rating": 0}
                count[method] = 0
            method_metrics[method]["fidelity"] += (metrics.get("fidelity", 0) or 0) * rel
            method_metrics[method]["simplicity"] += (metrics.get("simplicity", 0) or 0) * rel
            method_metrics[method]["stability"] += (metrics.get("stability", 0) or 0) * rel
            method_metrics[method]["rating"] += ratings.get(method, 3.0) * rel
            count[method] += rel

    estimated_scores = {}
    recommended_ai = None
    best_ai_score = -1
    recommended_method = None
    best_overall = -1e9

    if top_ids:
        top_entry = repository.get(top_ids[0], {})
        for ai_model, ai_res in top_entry.get("ai_results", {}).items():
            acc = ai_res.get("accuracy", 0) or 0
            if pd.notnull(acc) and acc > best_ai_score:
                best_ai_score = acc
                recommended_ai = ai_model

    # Ensure all modality-specific methods appear; supplement from repo if missing
    CORE_METHODS_BY_TYPE = {
        "tabular": {"SHAP", "LIME", "PFI", "PDP"},
        "text": {"LIME_Text", "SHAP_Text", "Integrated_Gradients_Text", "Attention_Weights", "Gradient_Text",
                 "InputXGradient_Text", "DeepLIFT_Text", "Occlusion_Text", "Shapley_Sampling_Text", "Rationale_Extraction"},
        "image": {"GradCAM", "GradCAM_pp", "Integrated_Gradients", "SmoothGrad", "LIME_Image", "SHAP_Image",
                  "Saliency_Maps", "Guided_Backprop", "Guided_GradCAM", "ScoreCAM", "LayerCAM", "Attention_Maps"},
        "timeseries": {"SHAP_TS", "LIME_TS", "Attention_TS", "Feature_Importance_TS", "Temporal_Attribution",
                       "Integrated_Gradients_TS", "Gradient_TS", "Permutation_Importance_TS"},
    }
    core_methods = CORE_METHODS_BY_TYPE.get(target_type, set())
    if core_methods and core_methods - set(method_metrics.keys()):
        for ds_id, e in repository.items():
            if not isinstance(ds_id, (int, str)) or ds_id in ("__survey_info__", "__domain_relevance__", "__dataset_relevance__"):
                continue
            raw = e.get("data_type")
            entry_dt = "tabular" if (raw is None or (hasattr(pd, "isna") and pd.isna(raw))) else str(raw).strip().lower()
            if not entry_dt or entry_dt == "nan":
                entry_dt = "tabular"
            if entry_dt != target_type:
                continue
            xai_results = e.get("xai_results", {})
            ratings = e.get("xai_method_ratings", {})
            for method in core_methods - set(method_metrics.keys()):
                if method not in xai_results:
                    continue
                metrics = xai_results[method]
                dataset_domain = e.get("domain")
                if dataset_domain is None or (hasattr(pd, "isna") and pd.isna(dataset_domain)):
                    dataset_domain = ""
                rel = get_dataset_relevance(
                    dataset_domain, domain, domain_relevance_map, dataset_relevance_map,
                    ds_id_a=dataset_id, ds_id_b=ds_id
                ) if use_relevance_weighting else 1.0
                if method not in method_metrics:
                    method_metrics[method] = {"fidelity": 0, "simplicity": 0, "stability": 0, "rating": 0}
                    count[method] = 0
                method_metrics[method]["fidelity"] += (metrics.get("fidelity", 0) or 0) * rel
                method_metrics[method]["simplicity"] += (metrics.get("simplicity", 0) or 0) * rel
                method_metrics[method]["stability"] += (metrics.get("stability", 0) or 0) * rel
                method_metrics[method]["rating"] += ratings.get(method, 3.0) * rel
                count[method] += rel

    for method, sums in method_metrics.items():
        n = count[method]
        if n == 0:
            continue
        avg_fid = sums["fidelity"] / n
        avg_simp = sums["simplicity"] / n
        avg_stab = sums["stability"] / n
        avg_rate = sums["rating"] / n
        domain_bonus = get_domain_xai_bonus(domain, method, survey_jsons)
        overall = (weights["fid"] * avg_fid + weights["stab"] * avg_stab +
                   weights["rate"] * avg_rate - weights["simp"] * (avg_simp / 100)) * domain_bonus
        estimated_scores[method] = {
            "avg_fidelity": avg_fid, "avg_simplicity": avg_simp, "avg_stability": avg_stab,
            "avg_rating": avg_rate, "domain_bonus": domain_bonus, "overall_score": overall
        }
        if overall > best_overall:
            best_overall = overall
            recommended_method = method

    return estimated_scores, recommended_method, recommended_ai, similar_datasets


def build_repository(data_df, results_ai_df, results_xai_df, ai_models_df, xai_models_df, survey_jsons=None):
    """
    Creates a nested dictionary keyed by dataset_id containing all information
    about datasets, AI models, and XAI methods.
    """
    # Build a lookup for XAI methods from the "xai models" sheet
    xai_lookup = {}
    for _, row in xai_models_df.iterrows():
        method_name = str(row["xai_model"]).strip()
        xai_lookup[method_name] = row.to_dict()

    # Initialize repository from dataset characteristics
    data_dict = {}
    for _, row in data_df.iterrows():
        ds_id = row["dataset_id"]
        fc = row.get("feature_count", 0)
        num = row.get("numeric_features", 0)
        cat = row.get("cat_features", 0)
        dtype_str = str(row.get("type", "") or "").lower()
        task_str = str(row.get("dataset_task", "") or "").lower()
        if (pd.isna(num) or pd.isna(cat)) and fc and (pd.notna(fc)):
            if "multivariate" in dtype_str or "tabular" in dtype_str:
                if "regression" in task_str or "classification" in task_str:
                    num = fc if pd.isna(num) else num
                    cat = 0 if pd.isna(cat) else cat
        data_dict[ds_id] = {
            "dataset_id": ds_id,
            "dataset_name": row.get("dataset_name", ""),
            "domain": row.get("domain", "").lower(),
            "data_type": row.get("data_type") if pd.notna(row.get("data_type")) else "tabular",
            "size": row.get("size", None),
            "type": row.get("type", ""),
            "task": row.get("dataset_task", ""),
            "feature_count": row.get("feature_count", None),
            "description": row.get("description", ""),
            "numeric_features": num if not pd.isna(num) else 0,
            "cat_features": cat if not pd.isna(cat) else 0,
            "NaN_values": 0 if pd.isna(row.get("NaN Values", row.get("NaN_values", 0))) else row.get("NaN Values", row.get("NaN_values", 0)),
            "image_width": row.get("image_width"),
            "image_height": row.get("image_height"),
            "channels": row.get("channels"),
            "num_classes": row.get("num_classes"),
            "series_length": row.get("series_length"),
            "num_channels": row.get("num_channels"),
            "avg_doc_length": row.get("avg_doc_length"),
            "vocab_size": row.get("vocab_size"),
            "max_length": row.get("max_length"),
            "ai_results": {},
            "xai_results": {}
        }

    # Merge AI performance from results_ai_df
    for _, row in results_ai_df.iterrows():
        ds_id = row["dataset_id"]
        if ds_id not in data_dict:
            data_type = None
            modality_fields = {}
            if "data_type" in data_df.columns:
                match = data_df[data_df["dataset_id"] == ds_id]
                if not match.empty:
                    r = match.iloc[0]
                    data_type = r.get("data_type", "tabular")
                    modality_fields = {
                        "image_width": r.get("image_width"), "image_height": r.get("image_height"),
                        "channels": r.get("channels"), "num_classes": r.get("num_classes"),
                        "series_length": r.get("series_length"), "num_channels": r.get("num_channels"),
                        "avg_doc_length": r.get("avg_doc_length"), "vocab_size": r.get("vocab_size"),
                        "max_length": r.get("max_length")
                    }
            data_dict[ds_id] = {"dataset_id": ds_id, "data_type": data_type or "tabular", "ai_results": {}, "xai_results": {}, **modality_fields}
        ai_model_name = row["ai_model_id"]
        accuracy = row.get("Accuracy", None)
        precision = row.get("Precision", row.get("Precision ", None))
        data_dict[ds_id]["ai_results"][ai_model_name] = {
            "accuracy": accuracy,
            "precision": precision
        }

    # Merge XAI quantitative metrics from results_xai_df
    for _, row in results_xai_df.iterrows():
        ds_id = row["Dataset ID"]
        if ds_id not in data_dict:
            data_type = None
            modality_fields = {}
            if "data_type" in data_df.columns:
                match = data_df[data_df["dataset_id"] == ds_id]
                if not match.empty:
                    r = match.iloc[0]
                    data_type = r.get("data_type", "tabular")
                    modality_fields = {
                        "image_width": r.get("image_width"), "image_height": r.get("image_height"),
                        "channels": r.get("channels"), "num_classes": r.get("num_classes"),
                        "series_length": r.get("series_length"), "num_channels": r.get("num_channels"),
                        "avg_doc_length": r.get("avg_doc_length"), "vocab_size": r.get("vocab_size"),
                        "max_length": r.get("max_length")
                    }
            data_dict[ds_id] = {"dataset_id": ds_id, "data_type": data_type or "tabular", "ai_results": {}, "xai_results": {}, **modality_fields}
        xai_method = str(row["XAI Method"]).strip()
        # Unify method names if needed
        if xai_method == "Permutation Feature Importance":
            xai_method = "PFI"
        elif xai_method == "Partial Dependence Plots":
            xai_method = "PDP"
        fidelity = row.get("Fidelity", None)
        simplicity = row.get("Simplicity", None)
        stability = row.get("Stability", None)
        xai_info = xai_lookup.get(xai_method, {})
        data_dict[ds_id]["xai_results"][xai_method] = {
            "fidelity": fidelity,
            "simplicity": simplicity,
            "stability": stability,
            "xai_model_info": xai_info
        }

    # Attach survey JSON info
    if survey_jsons is not None:
        data_dict["__survey_info__"] = survey_jsons

    return data_dict


def extract_features_from_new_dataset(df, data_type="tabular", metadata=None):
    """
    Extract relevant features from a user-uploaded dataset.
    Supports tabular, image, text, and timeseries data types.
    """
    size = df.shape[0]
    metadata = metadata or {}

    if data_type == "tabular":
        feature_count = df.shape[1]
        numeric_features = df.select_dtypes(include=np.number).shape[1]
        cat_features = df.select_dtypes(exclude=np.number).shape[1]
        total_elements = size * feature_count if size and feature_count else 1
        missing_count = df.isnull().sum().sum()
        missing_ratio = missing_count / total_elements if total_elements else 0
        return {
            "feature_count": feature_count,
            "size": size,
            "numeric_features": numeric_features,
            "cat_features": cat_features,
            "missing_ratio": missing_ratio if not pd.isnull(missing_ratio) else 0,
            "data_type": "tabular"
        }
    elif data_type == "image":
        num_classes = metadata.get("num_classes") or 0
        image_width = metadata.get("image_width") or 224
        image_height = metadata.get("image_height") or 224
        channels = metadata.get("channels") or 3
        if size == 0 and "size" in metadata:
            size = metadata.get("size", 0)
        return {
            "size": size,
            "num_classes": num_classes,
            "image_width": image_width,
            "image_height": image_height,
            "channels": channels,
            "data_type": "image"
        }
    elif data_type == "text":
        text_cols = df.select_dtypes(include=[object, "string"]).columns.tolist()
        if text_cols:
            lengths = df[text_cols[0]].astype(str).str.len()
            avg_doc_length = float(lengths.mean()) if len(lengths) > 0 else 0
            max_length = int(lengths.max()) if len(lengths) > 0 else 0
            all_words = " ".join(df[text_cols[0]].astype(str).tolist()).split()
            vocab_size = len(set(all_words)) if all_words else 0
        else:
            avg_doc_length = metadata.get("avg_doc_length") or 0
            max_length = metadata.get("max_length") or 0
            vocab_size = metadata.get("vocab_size") or 0
        return {
            "size": size,
            "avg_doc_length": avg_doc_length,
            "max_length": max_length,
            "vocab_size": vocab_size,
            "data_type": "text"
        }
    elif data_type == "timeseries":
        series_length = metadata.get("series_length")
        if series_length is None:
            series_length = df.shape[1] - 1 if df.shape[1] > 1 else df.shape[1]
        num_channels = metadata.get("num_channels") or 1
        num_classes = metadata.get("num_classes") or 0
        return {
            "size": size,
            "series_length": int(series_length),
            "num_channels": int(num_channels),
            "num_classes": num_classes,
            "data_type": "timeseries"
        }
    else:
        feature_count = df.shape[1]
        numeric_features = df.select_dtypes(include=np.number).shape[1]
        cat_features = df.select_dtypes(exclude=np.number).shape[1]
        total_elements = size * feature_count if size and feature_count else 1
        missing_count = df.isnull().sum().sum()
        missing_ratio = missing_count / total_elements if total_elements else 0
        return {
            "feature_count": feature_count,
            "size": size,
            "numeric_features": numeric_features,
            "cat_features": cat_features,
            "missing_ratio": missing_ratio if not pd.isnull(missing_ratio) else 0,
            "data_type": "tabular"
        }


def _domain_hash(domain):
    """Stable hash of domain string -> 0-1 for use in feature vector."""
    if not domain or (hasattr(pd, "isna") and pd.isna(domain)):
        return 0.0
    s = str(domain).strip().lower()
    h = sum(ord(c) * (i + 1) for i, c in enumerate(s)) % 10000
    return h / 10000.0


# Hybrid similarity: alpha for structural, (1-alpha) for semantic
SIMILARITY_STRUCTURAL_WEIGHT = 0.4

# Structural similarity metric: "cosine" (default), "euclidean", "manhattan"
SIMILARITY_METRIC = "cosine"


def _compute_similarity(vec_a, vec_b, metric=None):
    """
    Compute 0-1 similarity between two vectors.
    cosine: angle-based (default). euclidean/manhattan: penalize magnitude differences.
    """
    metric = metric or SIMILARITY_METRIC
    a = np.asarray(vec_a).flatten()
    b = np.asarray(vec_b).flatten()
    if metric == "cosine":
        sim = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]
        return float(np.clip(sim, 0, 1))
    if metric == "euclidean":
        dist = np.linalg.norm(a - b)
        denom = np.linalg.norm(a) + np.linalg.norm(b) + 1e-8
        norm_dist = dist / denom
        return float(np.clip(1.0 / (1.0 + norm_dist), 0, 1))
    if metric == "manhattan":
        dist = np.sum(np.abs(a - b))
        denom = np.sum(np.abs(a)) + np.sum(np.abs(b)) + 1e-8
        norm_dist = dist / denom
        return float(np.clip(1.0 / (1.0 + norm_dist), 0, 1))
    return float(np.clip(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0], 0, 1))


# Task type encoding for tabular: Classification=0, Regression=1, Clustering=2, Other=0.5
_TASK_ENCODING = {
    "classification": 0.0,
    "regression": 1.0,
    "clustering": 2.0,
    "causal-discovery": 0.5,
    "other": 0.5,
}


def _task_encode(task):
    """Map dataset_task to numeric value for feature vector."""
    if not task or (hasattr(pd, "isna") and pd.isna(task)):
        return 0.5
    s = str(task).strip().lower()
    for key, val in _TASK_ENCODING.items():
        if key in s:
            return val
    return 0.5


def extract_repo_features(entry):
    """
    Extract feature vector from a repository entry.
    Returns a fixed-dim vector for cosine similarity.
    Tabular: [size, fc, num, cat, missing_ratio, log_size, log_fc, domain_hash]
    - Extra dimensions (log-scale, domain) reduce spurious 1.0 similarities when
      many datasets have missing numeric_features/cat_features.
    """
    def safe_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    dt = str(entry.get("data_type") or "").strip().lower()
    sz = safe_float(entry.get("size", 0))

    if dt == "image":
        w = safe_float(entry.get("image_width", 0))
        h = safe_float(entry.get("image_height", 0))
        ch = safe_float(entry.get("channels", 0))
        complexity = w * h * ch if (w and h and ch) else 0
        num_classes = safe_float(entry.get("num_classes", 0))
        vector = np.array([sz, complexity, num_classes, 0.0, 0.0])
    elif dt == "text":
        avg_doc = safe_float(entry.get("avg_doc_length", 0))
        max_len = safe_float(entry.get("max_length", 0))
        vocab = safe_float(entry.get("vocab_size", 0))
        vector = np.array([sz, avg_doc, max_len, vocab, 0.0])
    elif dt == "timeseries":
        series_len = safe_float(entry.get("series_length", 0))
        num_ch = safe_float(entry.get("num_channels", 0))
        num_classes = safe_float(entry.get("num_classes", 0))
        vector = np.array([sz, series_len, num_ch, num_classes, 0.0])
    else:
        fc = safe_float(entry.get("feature_count", 0))
        num = safe_float(entry.get("numeric_features", 0))
        cat = safe_float(entry.get("cat_features", 0))
        total = fc * sz if fc and sz else 1.0
        nan_val = safe_float(entry.get("NaN_values", 0))
        missing_ratio = nan_val / total if total else 0
        log_sz = np.log1p(sz)
        log_fc = np.log1p(fc)
        domain = entry.get("domain") or ""
        domain_h = _domain_hash(domain)
        task = entry.get("task", entry.get("dataset_task", ""))
        task_h = _task_encode(task)
        vector = np.array([sz, fc, num, cat, missing_ratio, log_sz, log_fc, domain_h, task_h])

    return np.nan_to_num(vector, nan=0.0)


def get_domain_xai_bonus(domain, xai_method, survey_jsons):
    """
    Calculate domain-specific bonus factor for XAI methods.
    Maps modality-specific methods (SHAP_Text, LIME_TS, etc.) to base names for lookup.
    Uses built-in defaults when survey JSON is missing.
    """
    bonus = 1.0
    domain_lower = (domain or "").lower().strip()
    # Map modality-specific to base: SHAP_Text->SHAP, LIME_TS->LIME, etc.
    method_base = str(xai_method or "").split("_")[0].upper() if xai_method else ""

    try:
        # Built-in domain bonuses (used when survey missing or domain not in survey)
        if domain_lower in ["health and medicine", "healthcare", "health"]:
            bonus = 1.2 if method_base == "SHAP" else 0.9
        elif domain_lower in ["finance", "financial"]:
            bonus = 1.1 if method_base == "PDP" else 0.95
        elif domain_lower in ["manufacturing", "industry"]:
            bonus = 1.15 if method_base == "LIME" else 0.92
        elif domain_lower == "cybersecurity":
            bonus = 1.1 if method_base == "LIME" else 0.95
        elif domain_lower in ["autonomous_vehicles", "autonomous vehicles"]:
            bonus = 1.1 if method_base == "SHAP" else 0.95
        elif domain_lower in ["recommendation_systems", "recommendation systems"]:
            bonus = 1.05 if method_base == "PDP" else 0.98
        elif domain_lower in ["iot", "iot / sensors"]:
            bonus = 1.08 if method_base == "PFI" else 0.97
    except Exception:
        pass

    return bonus


def _features_to_vector(features, data_type, metadata=None):
    """Build vector from extracted features, matching extract_repo_features format."""
    def safe(v):
        try:
            return float(v) if v is not None else 0.0
        except (ValueError, TypeError):
            return 0.0

    dt = str(data_type or "tabular").strip().lower()
    sz = safe(features.get("size", 0))

    if dt == "image":
        w = safe(features.get("image_width", 0))
        h = safe(features.get("image_height", 0))
        ch = safe(features.get("channels", 0))
        complexity = w * h * ch if (w and h and ch) else 0
        num_classes = safe(features.get("num_classes", 0))
        return np.array([sz, complexity, num_classes, 0.0, 0.0])
    elif dt == "text":
        avg_doc = safe(features.get("avg_doc_length", 0))
        max_len = safe(features.get("max_length", 0))
        vocab = safe(features.get("vocab_size", 0))
        return np.array([sz, avg_doc, max_len, vocab, 0.0])
    elif dt == "timeseries":
        series_len = safe(features.get("series_length", 0))
        num_ch = safe(features.get("num_channels", 0))
        num_classes = safe(features.get("num_classes", 0))
        return np.array([sz, series_len, num_ch, num_classes, 0.0])
    else:
        fc = safe(features.get("feature_count", 0))
        num = safe(features.get("numeric_features", 0))
        cat = safe(features.get("cat_features", 0))
        miss = safe(features.get("missing_ratio", 0))
        log_sz = np.log1p(sz)
        log_fc = np.log1p(fc)
        domain_h = _domain_hash(features.get("domain", ""))
        task_val = features.get("task", (metadata or {}).get("task", ""))
        task_h = _task_encode(task_val)
        return np.array([sz, fc, num, cat, miss, log_sz, log_fc, domain_h, task_h])


def estimate_xai_score_for_new_dataset(new_df=None, repository=None, domain="general", top_k=3, weights=None, data_type="tabular", metadata=None, dataset_id=None, use_relevance_weighting=True):
    """
    Estimate XAI scores for a new dataset by finding similar datasets in the repository.
    Only compares within the same data_type (tabular, image, text, timeseries).
    If dataset_id is provided, uses that pre-loaded dataset's features (no new_df needed).
    """
    if weights is None:
        weights = {"fid": 0.4, "stab": 0.3, "rate": 0.2, "simp": 0.1}

    data_type = str(data_type or "tabular").strip().lower()

    if dataset_id is not None and repository is not None:
        entry = repository.get(dataset_id) or repository.get(str(dataset_id))
        if not entry and str(dataset_id).replace("-", "").isdigit():
            try:
                entry = repository.get(int(dataset_id))
            except (ValueError, TypeError):
                pass
        if entry:
            new_vector = extract_repo_features(entry)
        else:
            sample = next((e for k, e in repository.items() if isinstance(k, (int, str)) and isinstance(e, dict) and k not in ("__survey_info__", "__domain_relevance__", "__dataset_relevance__")), None)
            new_vector = np.zeros_like(extract_repo_features(sample)) if sample else np.zeros(9)
    else:
        new_features = extract_features_from_new_dataset(new_df, data_type=data_type, metadata=metadata)
        new_vector = _features_to_vector(new_features, data_type, metadata)
    new_vector = np.nan_to_num(new_vector, nan=0.0)

    # Only compare to entries matching the upload data_type (treat None/empty as tabular)
    target_type = data_type if data_type else "tabular"

    # Calculate similarity with matching datasets in repository (hybrid structural + semantic)
    dataset_relevance_map = repository.get("__dataset_relevance__") or {}
    alpha = SIMILARITY_STRUCTURAL_WEIGHT
    query_text = None
    if dataset_id is None and metadata:
        query_text = f"{metadata.get('dataset_name', 'uploaded')} {domain}"
    query_id = dataset_id if dataset_id is not None else "uploaded"
    similarities = []
    for ds_id, entry in repository.items():
        if not isinstance(ds_id, (int, str)) or ds_id in ("__survey_info__", "__domain_relevance__", "__dataset_relevance__"):
            continue
        raw = entry.get("data_type")
        entry_data_type = "tabular" if (raw is None or (hasattr(pd, "isna") and pd.isna(raw))) else str(raw).strip().lower()
        if not entry_data_type or entry_data_type == "nan":
            entry_data_type = "tabular"
        if entry_data_type != target_type:
            continue
        repo_vector = extract_repo_features(entry)
        structural_sim = _compute_similarity(new_vector, repo_vector)
        semantic_sim = compute_description_similarity(query_id, ds_id, repository, query_text=query_text)
        pair_rel = dataset_relevance_map.get((str(query_id), str(ds_id))) or dataset_relevance_map.get((str(ds_id), str(query_id)))
        if pair_rel is not None:
            semantic_sim = 0.7 * semantic_sim + 0.3 * float(pair_rel)
        final_sim = alpha * structural_sim + (1 - alpha) * semantic_sim
        similarities.append((ds_id, final_sim, entry.get("dataset_name", f"Dataset {ds_id}")))

    # Sort by similarity and get top-k
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    top_ids = [ds_id for ds_id, _, _ in similarities[:top_k]]

    # Return both similar datasets and their similarities
    similar_datasets = [(ds_id, sim, name) for ds_id, sim, name in similarities[:top_k]]

    # If no matching benchmarks, return empty recommendations
    if not similar_datasets:
        return {}, None, None, []

    # Aggregate metrics from similar datasets
    method_metrics = {}
    count = {}
    survey_jsons = repository.get("__survey_info__", {})
    domain_relevance_map = repository.get("__domain_relevance__") or {}
    dataset_relevance_map = repository.get("__dataset_relevance__") or {}
    if not use_relevance_weighting:
        domain_relevance_map = {}
        dataset_relevance_map = {}

    for ds_id in top_ids:
        entry = repository.get(ds_id, {})
        xai_results = entry.get("xai_results", {})
        ratings = entry.get("xai_method_ratings", {})
        dataset_domain = entry.get("domain")
        if dataset_domain is None or (hasattr(pd, "isna") and pd.isna(dataset_domain)):
            dataset_domain = ""
        rel = get_dataset_relevance(
            dataset_domain, domain, domain_relevance_map, dataset_relevance_map,
            ds_id_a=dataset_id, ds_id_b=ds_id
        ) if use_relevance_weighting else 1.0

        for method, metrics in xai_results.items():
            if method not in method_metrics:
                method_metrics[method] = {"fidelity": 0, "simplicity": 0, "stability": 0, "rating": 0}
                count[method] = 0

            method_metrics[method]["fidelity"] += (metrics.get("fidelity", 0) or 0) * rel
            method_metrics[method]["simplicity"] += (metrics.get("simplicity", 0) or 0) * rel
            method_metrics[method]["stability"] += (metrics.get("stability", 0) or 0) * rel
            method_metrics[method]["rating"] += ratings.get(method, 3.0) * rel
            count[method] += rel

    # Calculate final scores and apply domain bonus
    estimated_scores = {}
    recommended_ai = None
    best_ai_score = -1
    recommended_method = None
    best_overall = -1e9

    # Get best AI model (just use the one with highest accuracy from top similar dataset)
    if len(top_ids) > 0:
        top_entry = repository.get(top_ids[0], {})
        for ai_model, ai_res in top_entry.get("ai_results", {}).items():
            acc = ai_res.get("accuracy", 0) or 0
            if pd.notnull(acc) and acc > best_ai_score:
                best_ai_score = acc
                recommended_ai = ai_model

    # Ensure all modality-specific methods appear; supplement from repo if missing
    _CORE_BY_TYPE = {
        "tabular": {"SHAP", "LIME", "PFI", "PDP"},
        "text": {"LIME_Text", "SHAP_Text", "Integrated_Gradients_Text", "Attention_Weights", "Gradient_Text",
                 "InputXGradient_Text", "DeepLIFT_Text", "Occlusion_Text", "Shapley_Sampling_Text", "Rationale_Extraction"},
        "image": {"GradCAM", "GradCAM_pp", "Integrated_Gradients", "SmoothGrad", "LIME_Image", "SHAP_Image",
                  "Saliency_Maps", "Guided_Backprop", "Guided_GradCAM", "ScoreCAM", "LayerCAM", "Attention_Maps"},
        "timeseries": {"SHAP_TS", "LIME_TS", "Attention_TS", "Feature_Importance_TS", "Temporal_Attribution",
                       "Integrated_Gradients_TS", "Gradient_TS", "Permutation_Importance_TS"},
    }
    _core = _CORE_BY_TYPE.get(target_type, set())
    if _core and _core - set(method_metrics.keys()):
        for ds_id, e in repository.items():
            if not isinstance(ds_id, (int, str)) or ds_id in ("__survey_info__", "__domain_relevance__", "__dataset_relevance__"):
                continue
            raw = e.get("data_type")
            entry_dt = "tabular" if (raw is None or (hasattr(pd, "isna") and pd.isna(raw))) else str(raw).strip().lower()
            if not entry_dt or entry_dt == "nan":
                entry_dt = "tabular"
            if entry_dt != target_type:
                continue
            xai_results = e.get("xai_results", {})
            ratings = e.get("xai_method_ratings", {})
            for method in _core - set(method_metrics.keys()):
                if method not in xai_results:
                    continue
                metrics = xai_results[method]
                dataset_domain = e.get("domain")
                if dataset_domain is None or (hasattr(pd, "isna") and pd.isna(dataset_domain)):
                    dataset_domain = ""
                rel = get_dataset_relevance(
                    dataset_domain, domain, domain_relevance_map, dataset_relevance_map,
                    ds_id_a=dataset_id, ds_id_b=ds_id
                ) if use_relevance_weighting else 1.0
                if method not in method_metrics:
                    method_metrics[method] = {"fidelity": 0, "simplicity": 0, "stability": 0, "rating": 0}
                    count[method] = 0
                method_metrics[method]["fidelity"] += (metrics.get("fidelity", 0) or 0) * rel
                method_metrics[method]["simplicity"] += (metrics.get("simplicity", 0) or 0) * rel
                method_metrics[method]["stability"] += (metrics.get("stability", 0) or 0) * rel
                method_metrics[method]["rating"] += ratings.get(method, 3.0) * rel
                count[method] += rel

    # Calculate XAI method scores
    for method, sums in method_metrics.items():
        n = count[method]
        if n == 0:
            continue

        avg_fid = sums["fidelity"] / n
        avg_simp = sums["simplicity"] / n
        avg_stab = sums["stability"] / n
        avg_rate = sums["rating"] / n

        # Apply domain-specific bonus
        domain_bonus = get_domain_xai_bonus(domain, method, survey_jsons)

        overall = (weights["fid"] * avg_fid +
                   weights["stab"] * avg_stab +
                   weights["rate"] * avg_rate -
                   weights["simp"] * (avg_simp / 100)) * domain_bonus

        estimated_scores[method] = {
            "avg_fidelity": avg_fid,
            "avg_simplicity": avg_simp,
            "avg_stability": avg_stab,
            "avg_rating": avg_rate,
            "domain_bonus": domain_bonus,
            "overall_score": overall
        }

        if overall > best_overall:
            best_overall = overall
            recommended_method = method

    return estimated_scores, recommended_method, recommended_ai, similar_datasets


def get_method_description(method, repository):
    """Get XAI method description from repository."""
    survey_jsons = repository.get("__survey_info__") or {}
    if not isinstance(survey_jsons, dict):
        survey_jsons = {}
    method_descriptions = survey_jsons.get("json1", {}).get("xai_method_descriptions", {})

    default_descriptions = {
        "SHAP": "SHAP (SHapley Additive exPlanations) calculates feature importance by examining how each feature affects the model's prediction.",
        "LIME": "LIME (Local Interpretable Model-agnostic Explanations) builds a simpler model around a specific prediction to explain it.",
        "PFI": "PFI (Permutation Feature Importance) works by randomly shuffling a feature and measuring the decrease in model performance.",
        "PDP": "PDP (Partial Dependence Plots) shows the marginal effect of a feature on the predicted outcome.",
        # Image XAI
        "GradCAM": "GradCAM visualizes class-discriminative regions by combining gradients with feature maps.",
        "GradCAM_pp": "GradCAM++ improves GradCAM with better localization and multi-object handling.",
        "Integrated_Gradients": "Integrated Gradients attributes predictions by integrating gradients along a path from baseline to input.",
        "SmoothGrad": "SmoothGrad reduces noise in saliency maps by averaging gradients over noisy inputs.",
        "LIME_Image": "LIME for images explains by perturbing superpixels and fitting a local linear model.",
        "SHAP_Image": "SHAP applied to image inputs for pixel-wise importance attribution.",
        "Saliency_Maps": "Saliency maps highlight input regions that most influence the model output.",
        "Guided_Backprop": "Guided Backprop backpropagates only positive gradients to highlight activating features.",
        "Guided_GradCAM": "Guided GradCAM combines GradCAM with Guided Backprop for finer-grained visualizations.",
        "ScoreCAM": "ScoreCAM uses forward activation maps and their importance scores, no gradients needed.",
        "LayerCAM": "LayerCAM creates CAM from any layer with pixel-level localization.",
        "Attention_Maps": "Attention maps from transformer or attention layers visualize where the model focuses.",
        # Text XAI
        "LIME_Text": "LIME for text by masking tokens and measuring their impact on predictions.",
        "SHAP_Text": "SHAP for text models attributes importance to tokens or subwords.",
        "Integrated_Gradients_Text": "Integrated Gradients applied to text embeddings for token attribution.",
        "Attention_Weights": "Attention weights from transformer models show which tokens influence outputs.",
        "Gradient_Text": "Gradient-based attribution for text via gradient magnitude per token.",
        "InputXGradient_Text": "Input×Gradient attribution for text inputs.",
        "DeepLIFT_Text": "DeepLIFT assigns importance by comparing activations to a reference input.",
        "Occlusion_Text": "Occlusion measures importance by replacing tokens and observing prediction change.",
        "Shapley_Sampling_Text": "Shapley value estimation via sampling for text feature importance.",
        "Rationale_Extraction": "Rationale extraction identifies minimal token spans that justify predictions.",
        # Time series XAI
        "SHAP_TS": "SHAP for time series attributes importance to timesteps or features.",
        "LIME_TS": "LIME for time series explains by perturbing segments and fitting local models.",
        "Attention_TS": "Attention mechanisms in time series models show temporal importance.",
        "Feature_Importance_TS": "Feature importance for time series models (e.g., tree-based).",
        "Temporal_Attribution": "Temporal attribution assigns importance across time steps.",
        "Integrated_Gradients_TS": "Integrated Gradients for time series inputs.",
        "Gradient_TS": "Gradient-based attribution for time series.",
        "Permutation_Importance_TS": "Permutation importance for time series features.",
    }

    return method_descriptions.get(method, default_descriptions.get(method, f"No description available for {method}"))


def get_model_description(model, repository):
    """Get AI model description from repository."""
    survey_jsons = repository.get("__survey_info__", {})
    model_descriptions = survey_jsons.get("json2", {}).get("ai_model_descriptions", {})

    default_descriptions = {
        "Random Forest": "An ensemble learning method that operates by constructing multiple decision trees.",
        "XGBoost": "A scalable tree boosting system that uses gradient boosting framework.",
        "SVM": "Support Vector Machines are supervised learning models used for classification and regression analysis.",
        "Neural Network": "A series of algorithms that attempts to recognize underlying relationships in a set of data through a process that mimics how the human brain operates.",
        # Image models
        "ResNet18": "18-layer residual network for image classification with skip connections.",
        "ResNet50": "50-layer residual network, deeper variant for higher accuracy.",
        "MobileNetV2": "Lightweight CNN for mobile and edge deployment.",
        "EfficientNet_B0": "EfficientNet balances depth, width, and resolution for better accuracy-efficiency tradeoff.",
        "ViT_Base": "Vision Transformer applies transformer architecture to image patches.",
        "DenseNet121": "Densely connected convolutional network with feature reuse.",
        "InceptionV3": "Inception architecture with optimized Inception modules.",
        "VGG16": "16-layer VGG network with simple 3x3 convolutional blocks.",
        "SimpleCNN": "Lightweight convolutional neural network for image classification.",
        "ShuffleNetV2": "Efficient architecture using channel shuffle operations.",
        # Text models
        "BERT_base": "Bidirectional Encoder Representations from Transformers for NLP.",
        "DistilBERT": "Distilled BERT with fewer parameters and faster inference.",
        "RoBERTa": "Robustly optimized BERT pretraining approach.",
        "ALBERT": "A Lite BERT with parameter sharing for efficiency.",
        "BiLSTM": "Bidirectional LSTM for sequence modeling.",
        "LSTM": "Long Short-Term Memory network for sequential data.",
        "CNN_Text": "Convolutional neural network for text classification.",
        "Transformer_small": "Small transformer model for text tasks.",
        "GPT2_small": "Generative pre-trained transformer for text generation and understanding.",
        "ELECTRA": "Efficiently learning an encoder that classifies token replacements.",
        # Time series models
        "LSTM_TS": "LSTM for time series forecasting and classification.",
        "BiLSTM_TS": "Bidirectional LSTM for time series.",
        "GRU_TS": "Gated Recurrent Unit for time series modeling.",
        "Transformer_TS": "Transformer adapted for time series.",
        "TCN": "Temporal Convolutional Network for sequence modeling.",
        "CNN_1D": "1D convolutional network for time series.",
        "XGBoost_TS": "XGBoost applied to time series features.",
        "RandomForest_TS": "Random Forest for time series classification.",
    }

    return model_descriptions.get(model, default_descriptions.get(model, f"No description available for {model}"))


# Add rule-based response generator
def generate_rule_based_response(question, repository):
    """Generate rule-based responses based on repository data and recommendation results."""
    question_lower = question.lower()

    # Get recommendation results if available
    recommendation_results = st.session_state.get('recommendation_results')
    dataset_features = st.session_state.get('uploaded_dataset_features')

    # Check for questions about recommendations
    if recommendation_results and any(kw in question_lower for kw in
                                      ["recommend", "suggest", "best model", "best method", "which model",
                                       "which method"]):
        ai_model = recommendation_results['recommended_ai']
        xai_method = recommendation_results['recommended_xai']
        domain = recommendation_results['domain']

        return f"""## Recommendations for Your Dataset

Based on the analysis of your dataset ({recommendation_results['dataset_name']}), the system recommends:

### Recommended AI Model: {ai_model}
{get_model_description(ai_model, repository)}

### Recommended XAI Method: {xai_method}
{get_method_description(xai_method, repository)}

This recommendation is tailored for the {domain} domain and considers factors like:
- Fidelity (how accurately the XAI method represents the model's behavior)
- Stability (how consistent the explanations are)
- User Ratings (how well the method is rated for interpretability and trust)
- Domain-specific requirements

The recommendation was based on finding similar datasets in our benchmark repository and analyzing their performance with different AI and XAI combinations.
"""

    # Check for questions about seeing the data
    if any(kw in question_lower for kw in ["see my data", "view my data", "did you see", "my data", "uploaded data"]):
        if dataset_features:
            return f"""## Yes, I Can See Your Dataset

I can see that you've uploaded the dataset "{recommendation_results['dataset_name'] if recommendation_results else 'Unknown'}" with the following characteristics:

- **Feature Count:** {dataset_features['feature_count']}
- **Sample Size:** {dataset_features['size']} records
- **Numeric Features:** {dataset_features['numeric_features']}
- **Categorical Features:** {dataset_features['cat_features']}
- **Missing Values:** {dataset_features['missing_ratio']:.2%}

These characteristics were used to find similar benchmark datasets and generate appropriate recommendations.
"""
        else:
            return """## I Don't See Any Dataset Yet

It appears that you haven't uploaded a dataset yet, or the dataset information wasn't properly stored. Please go to the Recommendation System tab and:

1. Upload your dataset (CSV or Excel file)
2. Select your domain
3. Click "Generate Recommendations"

Once you've done this, I'll be able to see your dataset characteristics and provide specific recommendations.
"""

    # Check for questions about the dataset
    if dataset_features and any(
            kw in question_lower for kw in ["my dataset", "dataset characteristics", "dataset features"]):
        return f"""## Your Dataset Characteristics

Your uploaded dataset ({recommendation_results['dataset_name'] if recommendation_results else 'Unknown'}) has the following characteristics:

- **Feature Count:** {dataset_features['feature_count']}
- **Sample Size:** {dataset_features['size']}
- **Numeric Features:** {dataset_features['numeric_features']}
- **Categorical Features:** {dataset_features['cat_features']}
- **Missing Values:** {dataset_features['missing_ratio']:.2%}

These characteristics were used to find similar benchmark datasets and generate appropriate recommendations.
"""

    # Check for questions about specific XAI methods
    for method in ["SHAP", "LIME", "PFI", "PDP"]:
        if method.lower() in question_lower:
            description = get_method_description(method, repository)

            # Try to get method strengths and weaknesses
            strengths = []
            weaknesses = []
            survey_jsons = repository.get("__survey_info__", {})
            if "json1" in survey_jsons:
                strengths = survey_jsons["json1"].get("xai_method_strengths", {}).get(method, ["Easy to interpret"])
                weaknesses = survey_jsons["json1"].get("xai_method_weaknesses", {}).get(method, [
                    "Can be computationally expensive"])

            response = f"## {method}\n\n{description}\n\n"

            if strengths:
                response += "### Strengths\n"
                for strength in strengths[:3]:
                    response += f"- {strength}\n"

            if weaknesses:
                response += "\n### Weaknesses\n"
                for weakness in weaknesses[:3]:
                    response += f"- {weakness}\n"

            # Add comparison to recommended method if available
            if recommendation_results and recommendation_results['recommended_xai'] != method and method in \
                    recommendation_results['estimated_scores']:
                rec_method = recommendation_results['recommended_xai']
                rec_score = recommendation_results['estimated_scores'][rec_method]['overall_score']
                this_score = recommendation_results['estimated_scores'][method]['overall_score']

                response += f"\n### Comparison to Recommended Method\n"
                response += f"For your dataset, {method} received an overall score of {this_score:.2f} "
                response += f"compared to {rec_method}'s score of {rec_score:.2f}.\n"

                # Explain why the difference
                if this_score < rec_score:
                    response += f"\nThe main reasons {rec_method} was recommended over {method} for your data:\n"

                    # Check specific metrics
                    method_metrics = recommendation_results['estimated_scores'][method]
                    rec_metrics = recommendation_results['estimated_scores'][rec_method]

                    if rec_metrics['avg_fidelity'] > method_metrics['avg_fidelity']:
                        response += f"- {rec_method} has higher fidelity ({rec_metrics['avg_fidelity']:.2f} vs. {method_metrics['avg_fidelity']:.2f})\n"

                    if rec_metrics['avg_stability'] > method_metrics['avg_stability']:
                        response += f"- {rec_method} has better stability ({rec_metrics['avg_stability']:.2f} vs. {method_metrics['avg_stability']:.2f})\n"

                    if rec_metrics['domain_bonus'] > method_metrics['domain_bonus']:
                        response += f"- {rec_method} has a higher domain-specific bonus for {recommendation_results['domain']}\n"
                else:
                    response += f"\nAlthough {method} has a good score, {rec_method} was recommended due to other factors like domain-specific considerations."

            return response

    # Check for questions about AI models
    for model in ["Random Forest", "XGBoost", "SVM", "Neural Network"]:
        if model.lower() in question_lower or (model == "Neural Network" and "neural" in question_lower):
            description = get_model_description(model, repository)
            response = f"## {model}\n\n{description}\n\n"

            # Add comparison to recommended model if available
            if recommendation_results and recommendation_results['recommended_ai'] != model:
                response += f"\n### Comparison to Recommended Model\n"
                response += f"For your dataset, {recommendation_results['recommended_ai']} was recommended. "

                if model == "Random Forest":
                    response += "Random Forest is generally good for tabular data with mixed feature types and provides a balance between accuracy and interpretability."
                elif model == "XGBoost":
                    response += "XGBoost typically achieves higher accuracy but can be more prone to overfitting and is less interpretable."
                elif model == "SVM":
                    response += "SVM works well with clear margins of separation but may not perform as well on complex, high-dimensional datasets."
                elif model == "Neural Network":
                    response += "Neural Networks excel at capturing complex patterns but require more data and are less interpretable."

            return response

    # Check for questions about comparative performance
    if any(kw in question_lower for kw in ["compare", "versus", "vs", "difference between"]):
        if recommendation_results and all(
                method in recommendation_results['estimated_scores'] for method in ["SHAP", "LIME"]):
            shap_score = recommendation_results['estimated_scores']["SHAP"]
            lime_score = recommendation_results['estimated_scores']["LIME"]

            return f"""## Comparing SHAP and LIME for Your Dataset

For your dataset, here's how SHAP and LIME compare:

### SHAP
- Fidelity: {shap_score['avg_fidelity']:.2f}
- Stability: {shap_score['avg_stability']:.2f}
- User Rating: {shap_score['avg_rating']:.2f}
- Overall Score: {shap_score['overall_score']:.2f}

### LIME
- Fidelity: {lime_score['avg_fidelity']:.2f}
- Stability: {lime_score['avg_stability']:.2f}
- User Rating: {lime_score['avg_rating']:.2f}
- Overall Score: {lime_score['overall_score']:.2f}

### Key Differences
- SHAP provides global explanations (feature importance across the dataset) and local explanations (for individual predictions)
- LIME focuses on local explanations by creating a simpler model around a specific prediction
- SHAP has stronger theoretical guarantees based on game theory
- LIME can sometimes be more intuitive for non-technical users

The recommended method for your dataset was {recommendation_results['recommended_xai']}.
"""
        else:
            return """## Comparing XAI Methods

### SHAP vs. LIME
- **SHAP** provides both global and local explanations with strong theoretical foundations
- **LIME** creates locally interpretable models around specific predictions

### PFI vs. PDP
- **PFI** (Permutation Feature Importance) measures feature importance by shuffling feature values
- **PDP** (Partial Dependence Plots) shows how features affect predictions on average

For specific comparisons on your dataset, please generate recommendations in the Recommendation System tab first.
"""

    # Default response
    return """I can help you understand XAI methods and AI models. You can ask me about:

- Specific XAI methods like SHAP, LIME, PFI, or PDP
- AI models like Random Forest, XGBoost, SVM, or Neural Networks
- Domain-specific applications of XAI
- Your dataset characteristics
- The recommendations made by the system

Please be specific in your question so I can provide the most relevant information."""


###############################################################################
# UI Components
###############################################################################

def header_section():
    """Create the header section of the app."""
    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("//home/researcher/PycharmProjects/FAME_XAI_SCORING/humain-logo.png", width=150)

    with col2:
        st.title("XAI Model Recommendation System")
        st.markdown("""
        This system helps you choose the most appropriate AI and XAI models for your dataset based on your specific domain and requirements.
        Upload your dataset and get tailored recommendations with detailed explanations.
        """)


def sidebar_section():
    """Create the sidebar with API configuration."""
    st.sidebar.header("Data")
    st.sidebar.info("Benchmark data is loaded from the data/ folder (tabular, image, text, timeseries).")

    # API configuration section
    st.sidebar.markdown("---")
    st.sidebar.header("LLM API Configuration")

    # Check if API key is in session state
    if "openai_api_key" not in st.session_state:
        st.session_state["openai_api_key"] = ""

    # API key input with password masking
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=st.session_state["openai_api_key"],
        type="password",
        help="Enter your OpenAI API key for the XAI Assistant"
    )

    # Save API key to session state
    if api_key:
        st.session_state["openai_api_key"] = api_key

    # LLM model selection
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = "gpt-4"

    model_options = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
    selected_model = st.sidebar.selectbox(
        "LLM Model",
        options=model_options,
        index=model_options.index(st.session_state["llm_model"]),
        help="Select the LLM model to use for the XAI Assistant"
    )

    st.session_state["llm_model"] = selected_model

    # Add information on sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info("""
    This application provides recommendations for AI and XAI models based on your dataset characteristics and domain requirements.

    The system uses a repository of benchmark datasets, AI model results, and XAI method evaluations to make tailored recommendations.
    """)

    return None


def load_data(excel_file, json1_file, json2_file, json3_file, ratings_file):
    """Load all data sources and build the repository."""
    if not all([excel_file, ratings_file]):
        return None

    # Load Excel data
    excel_data = load_excel_data(excel_file.read())
    if excel_data is None:
        return None

    results_ai_df, results_xai_df, data_df, ai_models_df, xai_models_df = excel_data

    # Load JSON data
    survey_jsons = {}
    if json1_file:
        survey_jsons["json1"] = load_json_from_docx(json1_file.read())
    if json2_file:
        survey_jsons["json2"] = load_json_from_docx(json2_file.read())
    if json3_file:
        survey_jsons["json3"] = load_json_from_docx(json3_file.read())

    # Load ratings data
    ratings_df = load_qualitative_ratings(ratings_file.read())
    if ratings_df is None:
        return None

    # Build repository
    repository = build_repository(
        data_df, results_ai_df, results_xai_df,
        ai_models_df, xai_models_df,
        survey_jsons=survey_jsons
    )

    # Merge aggregated user ratings into each repository entry
    for _, row in ratings_df.iterrows():
        ds_id = row["dataset_id"]
        if ds_id in repository:
            # Include all available method ratings (legacy + extended)
            ratings = {}
            for col in ["SHAP", "LIME", "PFI", "PDP"]:
                if col in row and pd.notna(row[col]):
                    ratings[col] = float(row[col])
            for col in row.index:
                if col not in ("dataset_id", "SHAP", "LIME", "PFI", "PDP") and pd.notna(row.get(col)):
                    try:
                        ratings[col] = float(row[col])
                    except (ValueError, TypeError):
                        pass
            repository[ds_id]["xai_method_ratings"] = ratings

    # Save to session state
    st.session_state['repository'] = repository

    return repository


def recommendation_tab(repos):
    """Create the recommendation tab content. repos: {data_type: repository_dict}."""
    st.header("AI & XAI Model Recommendation")

    data_type_options = ["tabular", "image", "text", "timeseries"]
    selected_data_type = st.selectbox("Select your data type", data_type_options, format_func=lambda x: {"tabular": "Tabular (CSV/Excel)", "image": "Image", "text": "Text", "timeseries": "Time Series"}[x])

    repository = repos.get(selected_data_type) if isinstance(repos, dict) else repos
    datasets = get_available_datasets(selected_data_type, "data") if repository else []

    id_to_label = {str(d.get('dataset_id')): f"{d.get('dataset_name', d.get('dataset_id', ''))} ({d.get('domain', '')})" for d in datasets}
    dataset_ids = list(id_to_label.keys()) if id_to_label else []
    selected_dataset_id = st.selectbox(
        "Select benchmark dataset",
        options=dataset_ids,
        format_func=lambda x: id_to_label.get(str(x), str(x))
    ) if dataset_ids else None

    domain_options = [
        "General", "Healthcare", "Finance", "Manufacturing",
        "Retail", "Energy", "Transportation", "Education"
    ]
    selected_domain = st.selectbox("Select your domain", domain_options)
    st.session_state['selected_domain'] = selected_domain

    with st.expander("Advanced Options"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            w_fid = st.slider("Fidelity Weight", 0.1, 1.0, 0.4, 0.1)
        with col2:
            w_stab = st.slider("Stability Weight", 0.1, 1.0, 0.3, 0.1)
        with col3:
            w_rate = st.slider("User Rating Weight", 0.1, 1.0, 0.2, 0.1)
        with col4:
            w_simp = st.slider("Simplicity Weight", 0.1, 1.0, 0.1, 0.1)
        top_k = st.slider("Number of similar datasets to consider", 1, 10, 3)

    weights = {"fid": w_fid, "stab": w_stab, "rate": w_rate, "simp": w_simp}

    if st.button("Generate Recommendations", type="primary"):
        if not selected_dataset_id:
            st.warning("Please select a dataset.")
        elif repository is None:
            st.warning("No benchmark data for this type.")
        else:
            with st.spinner("Generating recommendations..."):
                try:
                    estimated_scores, recommended_xai, recommended_ai, similar_datasets = estimate_xai_score_for_dataset(
                        selected_dataset_id, repository, domain=selected_domain.lower(), top_k=top_k, weights=weights,
                        data_type=selected_data_type
                    )
                    entry = repository.get(selected_dataset_id, {})
                    features = {
                        'dataset_name': entry.get('dataset_name', str(selected_dataset_id)),
                        'domain': entry.get('domain', ''),
                        'data_type': selected_data_type
                    }
                    st.session_state['uploaded_dataset_features'] = features
                    st.session_state['recommendation_results'] = {
                        'dataset_name': entry.get('dataset_name', str(selected_dataset_id)),
                        'domain': selected_domain,
                        'recommended_ai': recommended_ai,
                        'recommended_xai': recommended_xai,
                        'estimated_scores': estimated_scores,
                        'similar_datasets': similar_datasets,
                        'weights': weights
                    }

                    # Display dataset info
                    st.subheader("Dataset Analysis")

                    if selected_data_type == "tabular":
                        cols = st.columns(5)
                        cols[0].metric("Feature Count", f"{features.get('feature_count', 'N/A')}")
                        cols[1].metric("Sample Size", f"{features.get('size', 'N/A')}")
                        cols[2].metric("Numeric Features", f"{features.get('numeric_features', 'N/A')}")
                        cols[3].metric("Categorical Features", f"{features.get('cat_features', 'N/A')}")
                        cols[4].metric("Missing Values", f"{features.get('missing_ratio', 0):.2%}")
                    elif selected_data_type == "image":
                        cols = st.columns(5)
                        cols[0].metric("Sample Size", f"{features.get('size', 'N/A')}")
                        cols[1].metric("Image Size", f"{features.get('image_width', '?')}x{features.get('image_height', '?')}")
                        cols[2].metric("Channels", f"{features.get('channels', 'N/A')}")
                        cols[3].metric("Num Classes", f"{features.get('num_classes', 'N/A')}")
                        cols[4].metric("Data Type", "Image")
                    elif selected_data_type == "text":
                        cols = st.columns(5)
                        cols[0].metric("Sample Size", f"{features.get('size', 'N/A')}")
                        cols[1].metric("Avg Doc Length", f"{features.get('avg_doc_length', 'N/A'):.0f}" if isinstance(features.get('avg_doc_length'), (int, float)) else "N/A")
                        cols[2].metric("Max Length", f"{features.get('max_length', 'N/A')}")
                        cols[3].metric("Vocab Size", f"{features.get('vocab_size', 'N/A')}")
                        cols[4].metric("Data Type", "Text")
                    else:
                        cols = st.columns(5)
                        cols[0].metric("Sample Size", f"{features.get('size', 'N/A')}")
                        cols[1].metric("Series Length", f"{features.get('series_length', 'N/A')}")
                        cols[2].metric("Num Channels", f"{features.get('num_channels', 'N/A')}")
                        cols[3].metric("Num Classes", f"{features.get('num_classes', 'N/A')}")
                        cols[4].metric("Data Type", "Time Series")

                    # Show similar datasets
                    st.subheader("Similar Benchmark Datasets")
                    if not similar_datasets:
                        st.info("No benchmark datasets found for this data type.")
                    else:
                        similar_df = pd.DataFrame([
                            {"Dataset ID": ds_id,
                             "Dataset Name": name,
                             "Similarity Score": f"{sim:.2f}"}
                            for ds_id, sim, name in similar_datasets
                        ])
                        st.dataframe(similar_df, use_container_width=True)

                    # Display recommendations
                    st.markdown("---")
                    st.subheader("Recommendations")

                    # AI Model recommendation
                    ai_col, xai_col = st.columns(2)

                    with ai_col:
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h3>Recommended AI Model</h3>
                            <h2 style="color: #4e89ae;">{recommended_ai or "Not available"}</h2>
                            <p>{get_model_description(recommended_ai, repository) if recommended_ai else "No benchmark data for this data type."}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with xai_col:
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h3>Recommended XAI Method</h3>
                            <h2 style="color: #4e89ae;">{recommended_xai or "Not available"}</h2>
                            <p>{get_method_description(recommended_xai, repository) if recommended_xai else "No benchmark data for this data type."}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # XAI Methods Comparison (only when we have results)
                    if estimated_scores:
                        st.markdown("---")
                        st.subheader("XAI Methods Comparison")

                        # Create a DataFrame for visualization
                        methods = list(estimated_scores.keys())
                        metrics = ["avg_fidelity", "avg_stability", "avg_rating", "overall_score"]
                        metric_names = ["Fidelity", "Stability", "User Rating", "Overall Score"]

                        # Bar chart for overall scores
                        scores_df = pd.DataFrame({
                            "Method": methods,
                            "Overall Score": [estimated_scores[m]["overall_score"] for m in methods]
                        })

                        fig = px.bar(
                            scores_df,
                            x="Method",
                            y="Overall Score",
                            color="Method",
                            title="XAI Methods Overall Scores"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Radar chart for comparing all metrics
                        radar_data = []
                        for method in methods:
                            method_data = {"Method": method}
                            for metric, name in zip(metrics[:-1], metric_names[:-1]):  # Exclude overall score
                                method_data[name] = estimated_scores[method][metric]
                            radar_data.append(method_data)

                        radar_df = pd.DataFrame(radar_data)

                        fig = px.line_polar(
                            radar_df,
                            r=[radar_df[col].values for col in radar_df.columns if col != "Method"],
                            theta=[col for col in radar_df.columns if col != "Method"],
                            line_close=True,
                            color="Method",
                            title="XAI Methods Comparison"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Detailed metrics table
                        st.subheader("Detailed Metrics")
                        detail_data = []
                        for method in methods:
                            method_data = {"Method": method}
                            for metric, name in zip(metrics, metric_names):
                                method_data[name] = f"{estimated_scores[method][metric]:.3f}"
                            method_data["Domain Bonus"] = f"{estimated_scores[method]['domain_bonus']:.2f}"
                            detail_data.append(method_data)

                        detail_df = pd.DataFrame(detail_data)
                        st.dataframe(detail_df, use_container_width=True)

                except Exception as e:
                    st.error(f"Error processing dataset: {e}")


def chatbot_tab(repository):
    """Create the chatbot tab content with LLM integration and fallback to rule-based responses."""
    st.header("XAI Assistant")

    # Initialize LLM assistant
    if "llm_assistant" not in st.session_state:
        api_key = st.session_state.get("openai_api_key", "")
        st.session_state["llm_assistant"] = LLMXAIAssistant(api_key=api_key)

    # Update assistant if API key changes
    current_api_key = st.session_state.get("openai_api_key", "")
    current_model = st.session_state.get("llm_model", "gpt-4")
    if current_api_key != st.session_state["llm_assistant"].api_key:
        st.session_state["llm_assistant"] = LLMXAIAssistant(api_key=current_api_key)

    # Update model if changed
    st.session_state["llm_assistant"].model = current_model

    # Display repository and recommendation info for debugging
    with st.expander("Debug Information", expanded=False):
        st.write("Repository available:", repository is not None)
        st.write("Recommendation results available:", st.session_state.get('recommendation_results') is not None)
        if st.session_state.get('recommendation_results'):
            st.json(st.session_state.get('recommendation_results'))

    # Show info about recommendation results if available
    if st.session_state.get('recommendation_results'):
        with st.expander("Recommendation Results Available", expanded=True):
            results = st.session_state['recommendation_results']
            st.markdown(f"""
            **Dataset:** {results['dataset_name']}  
            **Domain:** {results['domain']}  
            **Recommended AI Model:** {results['recommended_ai']}  
            **Recommended XAI Method:** {results['recommended_xai']}

            You can ask me about these recommendations!
            """)
    else:
        st.info(
            "No recommendation results available yet. Please go to the 'Recommendation System' tab to upload your dataset and generate recommendations.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
             "content": "Hello! I'm your XAI Assistant. I can help you understand AI and XAI models, their applications, and how to choose the right ones for your needs. What would you like to know?"}
        ]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if user_input := st.chat_input("Ask me about XAI methods, AI models, or domain-specific applications..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.spinner("Thinking..."):
            if repository is None:
                response = "Please upload the necessary data files in the sidebar to enable full chatbot functionality."
            elif not current_api_key:
                response = "Please enter an OpenAI API key in the sidebar to enable the LLM-based XAI Assistant. For now, I'll use a rule-based approach to answer your questions."
                response = generate_rule_based_response(user_input, repository)
            else:
                # Try using LLM assistant
                try:
                    response = st.session_state["llm_assistant"].generate_response(user_input, repository)
                except Exception as e:
                    # Fall back to rule-based responses if API fails
                    st.error(f"LLM API error: {str(e)}. Using fallback response mechanism.")
                    response = generate_rule_based_response(user_input, repository)

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


def resources_tab():
    """Create the resources tab with documentation."""
    st.header("Resources & Documentation")

    # Create tabs for different resource sections
    doc_tabs = st.tabs(["XAI Methods", "AI Models", "Using the System", "References"])

    with doc_tabs[0]:
        st.subheader("XAI Methods Overview")

        methods = {
            "SHAP (SHapley Additive exPlanations)": {
                "description": "SHAP assigns each feature an importance value for a particular prediction. It's based on game theory and provides both local and global interpretability.",
                "strengths": ["Solid theoretical foundation", "Can handle any ML model",
                              "Provides both local and global explanations"],
                "weaknesses": ["Computationally expensive for large datasets",
                               "Can be difficult to interpret for complex interactions",
                               "May not capture all dependencies"]
            },
            "LIME (Local Interpretable Model-agnostic Explanations)": {
                "description": "LIME explains the predictions of any classifier by approximating it locally with an interpretable model.",
                "strengths": ["Model-agnostic", "Intuitive explanations", "Works well for both text and images"],
                "weaknesses": ["Unstable explanations", "Limited to local explanations",
                               "Sensitive to sampling parameters"]
            },
            "PFI (Permutation Feature Importance)": {
                "description": "PFI measures the importance of a feature by calculating the increase in the model's prediction error after permuting the feature values.",
                "strengths": ["Simple to implement", "Intuitive interpretation", "Model-agnostic"],
                "weaknesses": ["Can be misleading with correlated features", "Needs a performance metric",
                               "Cannot detect interaction effects"]
            },
            "PDP (Partial Dependence Plots)": {
                "description": "PDPs show the marginal effect of a feature on the predicted outcome, averaged across all other features.",
                "strengths": ["Visualizes feature relationships", "Works with any model", "Shows non-linear effects"],
                "weaknesses": ["Assumes feature independence", "Can be computationally expensive",
                               "May hide heterogeneous effects"]
            }
        }

        for method, info in methods.items():
            with st.expander(method):
                st.markdown(f"**Description**: {info['description']}")

                st.markdown("**Strengths**:")
                for strength in info["strengths"]:
                    st.markdown(f"- {strength}")

                st.markdown("**Weaknesses**:")
                for weakness in info["weaknesses"]:
                    st.markdown(f"- {weakness}")

    with doc_tabs[1]:
        st.subheader("AI Models Overview")

        models = {
            "Random Forest": {
                "description": "An ensemble learning method that operates by constructing multiple decision trees during training and outputting the class that is the mode of the classes of the individual trees.",
                "best_for": ["Tabular data", "Classification and regression", "High-dimensional datasets"],
                "limitations": ["Less effective for very high-dimensional sparse data",
                                "Can be computationally intensive", "Black-box nature"]
            },
            "XGBoost": {
                "description": "A decision-tree-based ensemble machine learning algorithm that uses a gradient boosting framework, known for its performance and speed.",
                "best_for": ["Structured/tabular data", "Competitions", "When performance is critical"],
                "limitations": ["Can overfit on noisy data", "Requires careful tuning",
                                "Less interpretable than simpler models"]
            },
            "SVM (Support Vector Machine)": {
                "description": "A supervised learning model that analyzes data for classification and regression analysis, using a technique that finds the hyperplane that best separates classes.",
                "best_for": ["High-dimensional spaces", "Text classification",
                             "When clear margin of separation exists"],
                "limitations": ["Poor performance with overlapping classes", "Sensitive to kernel choice",
                                "Computationally intensive for large datasets"]
            },
            "Neural Networks": {
                "description": "Computational models inspired by the human brain, consisting of layers of interconnected nodes that can learn complex patterns in data.",
                "best_for": ["Image and speech recognition", "Natural language processing",
                             "Complex pattern recognition"],
                "limitations": ["Require large amounts of data", "Computationally expensive",
                                "Limited interpretability"]
            }
        }

        for model, info in models.items():
            with st.expander(model):
                st.markdown(f"**Description**: {info['description']}")

                st.markdown("**Best For**:")
                for use_case in info["best_for"]:
                    st.markdown(f"- {use_case}")

                st.markdown("**Limitations**:")
                for limitation in info["limitations"]:
                    st.markdown(f"- {limitation}")

    with doc_tabs[2]:
        st.subheader("Using the Recommendation System")

        st.markdown("""
        ### Step 1: Upload Data Files

        Before using the system, upload the required data files in the sidebar:
        - Excel file with benchmark data (required)
        - JSON files with method and model descriptions (optional but recommended)
        - XAI Ratings CSV (required)

        ### Step 2: Upload Your Dataset

        In the Recommendation tab, upload your dataset (CSV or Excel) for analysis.

        ### Step 3: Select Your Domain

        Choose the domain that best matches your application context.

        ### Step 4: Configure Advanced Options (Optional)

        Adjust the weights for different metrics based on your priorities:
        - Fidelity: How accurately the XAI method represents the model's behavior
        - Stability: How consistent the explanations are with small input changes
        - User Rating: How users have rated the method for interpretability and trust
        - Simplicity: How easy the method's outputs are to understand

        ### Step 5: Generate Recommendations

        Click "Generate Recommendations" to get tailored AI and XAI model suggestions.

        ### Step 6: Explore Results

        Review the recommendations, compare methods, and see detailed metrics.

        ### Step 7: Ask the XAI Assistant

        Use the Chatbot tab to ask specific questions about methods, models, or applications.
        """)

    with doc_tabs[3]:
        st.subheader("References & Further Reading")

        st.markdown("""
        ### Research Papers

        - Molnar, C. (2022). Interpretable Machine Learning. A Guide for Making Black Box Models Explainable.
        - Adadi, A., & Berrada, M. (2018). Peeking inside the black-box: A survey on Explainable Artificial Intelligence (XAI).
        - Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions.
        - Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier.

        ### Online Resources

        - [SHAP Documentation](https://shap.readthedocs.io/)
        - [LIME Documentation](https://lime-ml.readthedocs.io/)
        - [Scikit-learn Permutation Importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
        - [Partial Dependence Plots in Scikit-learn](https://scikit-learn.org/stable/modules/partial_dependence.html)

        ### Books

        - Interpretable Machine Learning by Christoph Molnar
        - Explanatory Model Analysis by Przemyslaw Biecek and Tomasz Burzykowski
        - Practical Explainable AI Using Python by Akshay Kumar Budhkar
        """)


def main():
    """Main application function."""
    header_section()
    sidebar_section()

    # Load data from data/ folders
    repos = st.session_state.get('repository_by_type')
    if repos is None:
        repos = load_data_from_folders("data")
        st.session_state['repository_by_type'] = repos
    if not repos:
        st.warning("No benchmark data found. Run scripts/data_management/revise_and_split_data.py to generate the data/ folder.")
        st.stop()

    # Merge all repos for chatbot (needs full context)
    merged_repo = {}
    for repo in repos.values():
        merged_repo.update({k: v for k, v in repo.items() if k not in ("__survey_info__", "__domain_relevance__", "__dataset_relevance__")})
    if repos:
        first = next(iter(repos.values()))
        if "__survey_info__" in first:
            merged_repo["__survey_info__"] = first["__survey_info__"]
        if "__domain_relevance__" in first:
            merged_repo["__domain_relevance__"] = first["__domain_relevance__"]
        if "__dataset_relevance__" in first:
            merged_repo["__dataset_relevance__"] = first["__dataset_relevance__"]

    tabs = st.tabs(["Recommendation System", "XAI Assistant", "Resources"])

    with tabs[0]:
        recommendation_tab(repos)

    with tabs[1]:
        chatbot_tab(merged_repo)

    with tabs[2]:
        resources_tab()


if __name__ == "__main__":
    main()
