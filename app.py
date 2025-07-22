import streamlit as st
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
            if not isinstance(ds_id, (int, str)) or ds_id == "__survey_info__":
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


def load_qualitative_ratings(file_bytes):
    """Load CSV containing XAI method ratings."""
    try:
        df = pd.read_csv(BytesIO(file_bytes))

        required_cols = [
            "dataset_id",
            "interpretability_SHAP", "interpretability_LIME", "interpretability_PFI", "interpretability_PDP",
            "understanding_SHAP", "understanding_LIME", "understanding_PFI", "understanding_PDP",
            "trust_SHAP", "trust_LIME", "trust_PFI", "trust_PDP"
        ]

        for col in required_cols:
            if col not in df.columns:
                st.error(f"Column '{col}' not found in ratings CSV.")
                return None

        df["SHAP"] = (df["interpretability_SHAP"] + df["understanding_SHAP"] + df["trust_SHAP"]) / 3.0
        df["LIME"] = (df["interpretability_LIME"] + df["understanding_LIME"] + df["trust_LIME"]) / 3.0
        df["PFI"] = (df["interpretability_PFI"] + df["understanding_PFI"] + df["trust_PFI"]) / 3.0
        df["PDP"] = (df["interpretability_PDP"] + df["understanding_PDP"] + df["trust_PDP"]) / 3.0

        grouped = df.groupby("dataset_id")[["SHAP", "LIME", "PFI", "PDP"]].mean().reset_index()
        return grouped

    except Exception as e:
        st.error(f"Error loading ratings CSV: {e}")
        return None


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
        data_dict[ds_id] = {
            "dataset_id": ds_id,
            "dataset_name": row.get("dataset_name", ""),
            "domain": row.get("domain", "").lower(),
            "size": row.get("size", None),
            "type": row.get("type", ""),
            "task": row.get("dataset_task", ""),
            "feature_count": row.get("feature_count", None),
            "description": row.get("description", ""),
            "numeric_features": row.get("numeric_features", 0),
            "cat_features": row.get("cat_features", 0),
            "NaN_values": row.get("NaN Values", 0),
            "ai_results": {},
            "xai_results": {}
        }

    # Merge AI performance from results_ai_df
    for _, row in results_ai_df.iterrows():
        ds_id = row["dataset_id"]
        if ds_id not in data_dict:
            data_dict[ds_id] = {"dataset_id": ds_id, "ai_results": {}, "xai_results": {}}
        ai_model_name = row["ai_model_id"]
        accuracy = row.get("Accuracy", None)
        precision = row.get("Precision", None)
        data_dict[ds_id]["ai_results"][ai_model_name] = {
            "accuracy": accuracy,
            "precision": precision
        }

    # Merge XAI quantitative metrics from results_xai_df
    for _, row in results_xai_df.iterrows():
        ds_id = row["Dataset ID"]
        if ds_id not in data_dict:
            data_dict[ds_id] = {"dataset_id": ds_id, "ai_results": {}, "xai_results": {}}
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


def extract_features_from_new_dataset(df):
    """
    Extract relevant features from a user-uploaded dataset.
    """
    feature_count = df.shape[1]
    size = df.shape[0]
    numeric_features = df.select_dtypes(include=np.number).shape[1]
    cat_features = df.select_dtypes(exclude=np.number).shape[1]
    total_elements = size * feature_count if size and feature_count else 1
    missing_count = df.isnull().sum().sum()
    missing_ratio = missing_count / total_elements
    return {
        "feature_count": feature_count,
        "size": size,
        "numeric_features": numeric_features,
        "cat_features": cat_features,
        "missing_ratio": missing_ratio if not pd.isnull(missing_ratio) else 0
    }


def extract_repo_features(entry):
    """
    Extract feature vector from a repository entry.
    """

    def safe_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    fc = safe_float(entry.get("feature_count", 0))
    sz = safe_float(entry.get("size", 0))
    num = safe_float(entry.get("numeric_features", 0))
    cat = safe_float(entry.get("cat_features", 0))
    total = fc * sz if fc and sz else 1.0
    nan_val = safe_float(entry.get("NaN_values", 0))
    missing_ratio = nan_val / total
    vector = np.array([fc, sz, num, cat, missing_ratio])
    return np.nan_to_num(vector, nan=0.0)


def get_domain_xai_bonus(domain, xai_method, survey_jsons):
    """
    Calculate domain-specific bonus factor for XAI methods.
    """
    bonus = 1.0
    try:
        # Use survey JSON data to determine bonuses
        usage = survey_jsons.get("json3", {}).get("xai_usage_in_industries", {})
        domain_lower = domain.lower()

        if domain_lower in usage:
            # Domain-specific bonuses
            if domain_lower in ["health and medicine", "healthcare", "health"]:
                if xai_method.upper() == "SHAP":
                    bonus = 1.2
                else:
                    bonus = 0.9
            elif domain_lower in ["finance", "financial"]:
                if xai_method.upper() == "PDP":
                    bonus = 1.1
                else:
                    bonus = 0.95
            elif domain_lower in ["manufacturing", "industry"]:
                if xai_method.upper() == "LIME":
                    bonus = 1.15
                else:
                    bonus = 0.92
    except Exception as e:
        pass

    return bonus


def estimate_xai_score_for_new_dataset(new_df, repository, domain="general", top_k=3, weights=None):
    """
    Estimate XAI scores for a new dataset by finding similar datasets in the repository.
    """
    if weights is None:
        weights = {"fid": 0.4, "stab": 0.3, "rate": 0.2, "simp": 0.1}

    new_features = extract_features_from_new_dataset(new_df)
    new_vector = np.array([new_features["feature_count"],
                           new_features["size"],
                           new_features["numeric_features"],
                           new_features["cat_features"],
                           new_features["missing_ratio"]])
    new_vector = np.nan_to_num(new_vector, nan=0.0)

    # Calculate similarity with all datasets in repository
    similarities = []
    for ds_id, entry in repository.items():
        if not isinstance(ds_id, (int, str)) or ds_id == "__survey_info__":
            continue
        repo_vector = extract_repo_features(entry)
        sim = cosine_similarity(new_vector.reshape(1, -1), repo_vector.reshape(1, -1))[0][0]
        similarities.append((ds_id, sim, entry.get("dataset_name", f"Dataset {ds_id}")))

    # Sort by similarity and get top-k
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    top_ids = [ds_id for ds_id, _, _ in similarities[:top_k]]

    # Return both similar datasets and their similarities
    similar_datasets = [(ds_id, sim, name) for ds_id, sim, name in similarities[:top_k]]

    # Aggregate metrics from similar datasets
    method_metrics = {}
    count = {}
    survey_jsons = repository.get("__survey_info__", {})

    for ds_id in top_ids:
        entry = repository.get(ds_id, {})
        xai_results = entry.get("xai_results", {})
        ratings = entry.get("xai_method_ratings", {})

        for method, metrics in xai_results.items():
            if method not in method_metrics:
                method_metrics[method] = {"fidelity": 0, "simplicity": 0, "stability": 0, "rating": 0}
                count[method] = 0

            method_metrics[method]["fidelity"] += metrics.get("fidelity", 0) or 0
            method_metrics[method]["simplicity"] += metrics.get("simplicity", 0) or 0
            method_metrics[method]["stability"] += metrics.get("stability", 0) or 0
            method_metrics[method]["rating"] += ratings.get(method, 3.0)
            count[method] += 1

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
    survey_jsons = repository.get("__survey_info__", {})
    method_descriptions = survey_jsons.get("json1", {}).get("xai_method_descriptions", {})

    default_descriptions = {
        "SHAP": "SHAP (SHapley Additive exPlanations) calculates feature importance by examining how each feature affects the model's prediction.",
        "LIME": "LIME (Local Interpretable Model-agnostic Explanations) builds a simpler model around a specific prediction to explain it.",
        "PFI": "PFI (Permutation Feature Importance) works by randomly shuffling a feature and measuring the decrease in model performance.",
        "PDP": "PDP (Partial Dependence Plots) shows the marginal effect of a feature on the predicted outcome.",
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
    """Create the sidebar with file upload options and API configuration."""
    st.sidebar.header("Data Sources")

    # File upload section
    excel_file = st.sidebar.file_uploader("Upload Excel file with benchmark data", type=["xlsx"])

    json1_file = st.sidebar.file_uploader("Upload JSON 1 (Method Descriptions)", type=["docx"])
    json2_file = st.sidebar.file_uploader("Upload JSON 2 (Model Descriptions)", type=["docx"])
    json3_file = st.sidebar.file_uploader("Upload JSON 3 (Domain Knowledge)", type=["docx"])

    ratings_file = st.sidebar.file_uploader("Upload XAI Ratings CSV", type=["csv"])

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

    return excel_file, json1_file, json2_file, json3_file, ratings_file


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
            repository[ds_id]["xai_method_ratings"] = {
                "SHAP": row["SHAP"],
                "LIME": row["LIME"],
                "PFI": row["PFI"],
                "PDP": row["PDP"]
            }

    # Save to session state
    st.session_state['repository'] = repository

    return repository


def recommendation_tab(repository):
    """Create the recommendation tab content."""
    st.header("AI & XAI Model Recommendation")

    # Dataset upload
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

    # Domain selection
    domain_options = [
        "General", "Healthcare", "Finance", "Manufacturing",
        "Retail", "Energy", "Transportation", "Education"
    ]
    selected_domain = st.selectbox("Select your domain", domain_options)

    # Store selected domain in session state
    st.session_state['selected_domain'] = selected_domain

    # Advanced options
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
        if uploaded_file is None:
            st.warning("Please upload a dataset file.")
        elif repository is None:
            st.warning("Please upload the necessary benchmark data files.")
        else:
            with st.spinner("Analyzing dataset and generating recommendations..."):
                # Load the uploaded dataset
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)

                    # Extract dataset features and store in session state
                    features = extract_features_from_new_dataset(df)
                    st.session_state['uploaded_dataset_features'] = features

                    # Extract features and generate recommendations
                    estimated_scores, recommended_xai, recommended_ai, similar_datasets = estimate_xai_score_for_new_dataset(
                        df, repository, domain=selected_domain.lower(), top_k=top_k, weights=weights
                    )

                    # Store recommendation results in session state
                    st.session_state['recommendation_results'] = {
                        'dataset_name': uploaded_file.name,
                        'domain': selected_domain,
                        'recommended_ai': recommended_ai,
                        'recommended_xai': recommended_xai,
                        'estimated_scores': estimated_scores,
                        'similar_datasets': similar_datasets,
                        'weights': weights
                    }

                    # Display dataset info
                    st.subheader("Dataset Analysis")

                    cols = st.columns(5)
                    cols[0].metric("Feature Count", f"{features['feature_count']}")
                    cols[1].metric("Sample Size", f"{features['size']}")
                    cols[2].metric("Numeric Features", f"{features['numeric_features']}")
                    cols[3].metric("Categorical Features", f"{features['cat_features']}")
                    cols[4].metric("Missing Values", f"{features['missing_ratio']:.2%}")

                    # Show similar datasets
                    st.subheader("Similar Benchmark Datasets")
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
                            <p>{get_model_description(recommended_ai, repository)}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with xai_col:
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h3>Recommended XAI Method</h3>
                            <h2 style="color: #4e89ae;">{recommended_xai or "Not available"}</h2>
                            <p>{get_method_description(recommended_xai, repository)}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # XAI Methods Comparison
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
    # Page configuration
    header_section()

    # Sidebar for file uploads and API key
    excel_file, json1_file, json2_file, json3_file, ratings_file = sidebar_section()

    # Load data and build repository using session state for persistence
    repository = st.session_state.get('repository')
    if all([excel_file, ratings_file]):
        new_repository = load_data(excel_file, json1_file, json2_file, json3_file, ratings_file)
        if new_repository:
            repository = new_repository
            st.success("All data loaded successfully!")

    # Main content tabs
    tabs = st.tabs(["Recommendation System", "XAI Assistant", "Resources"])

    with tabs[0]:
        recommendation_tab(repository)

    with tabs[1]:
        chatbot_tab(repository)

    with tabs[2]:
        resources_tab()


if __name__ == "__main__":
    main()
