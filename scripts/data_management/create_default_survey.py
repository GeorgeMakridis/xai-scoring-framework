#!/usr/bin/env python3
"""
Create default survey JSON files when JSON_1/2/3.docx are not available.
Writes json1.json, json2.json, json3.json to data/shared/survey/.
"""

import json
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
SURVEY_DIR = os.path.join(DATA_ROOT, "shared", "survey")

# json1: XAI method descriptions, strengths, weaknesses
JSON1 = {
    "xai_method_descriptions": {
        "SHAP": "SHAP (SHapley Additive exPlanations) calculates feature importance by examining how each feature affects the model's prediction.",
        "LIME": "LIME (Local Interpretable Model-agnostic Explanations) builds a simpler model around a specific prediction to explain it.",
        "PFI": "PFI (Permutation Feature Importance) works by randomly shuffling a feature and measuring the decrease in model performance.",
        "PDP": "PDP (Partial Dependence Plots) shows the marginal effect of a feature on the predicted outcome.",
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
        "SHAP_TS": "SHAP for time series attributes importance to timesteps or features.",
        "LIME_TS": "LIME for time series explains by perturbing segments and fitting local models.",
        "Attention_TS": "Attention mechanisms in time series models show temporal importance.",
        "Feature_Importance_TS": "Feature importance for time series models (e.g., tree-based).",
        "Temporal_Attribution": "Temporal attribution assigns importance across time steps.",
        "Integrated_Gradients_TS": "Integrated Gradients for time series inputs.",
        "Gradient_TS": "Gradient-based attribution for time series.",
        "Permutation_Importance_TS": "Permutation importance for time series features.",
    },
    "xai_method_strengths": {
        "SHAP": ["Theoretically grounded", "Consistent attributions", "Local and global explanations"],
        "LIME": ["Model-agnostic", "Intuitive", "Fast for local explanations"],
        "PFI": ["Simple to implement", "Model-agnostic", "Global importance"],
        "PDP": ["Intuitive visualization", "Shows feature effects", "Model-agnostic"],
    },
    "xai_method_weaknesses": {
        "SHAP": ["Can be computationally expensive", "Kernel SHAP may be slow"],
        "LIME": ["Sensitive to sampling", "Local only", "May miss global patterns"],
        "PFI": ["Correlated features can distort", "Permutation can be slow"],
        "PDP": ["Assumes feature independence", "Can hide interactions"],
    },
}

# json2: AI model descriptions
JSON2 = {
    "ai_model_descriptions": {
        "Random Forest": "An ensemble learning method that operates by constructing multiple decision trees.",
        "XGBoost": "A scalable tree boosting system that uses gradient boosting framework.",
        "SVM": "Support Vector Machines are supervised learning models used for classification and regression analysis.",
        "Neural Network": "A series of algorithms that attempts to recognize underlying relationships in a set of data through a process that mimics how the human brain operates.",
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
        "LSTM_TS": "LSTM for time series forecasting and classification.",
        "BiLSTM_TS": "Bidirectional LSTM for time series.",
        "GRU_TS": "Gated Recurrent Unit for time series modeling.",
        "Transformer_TS": "Transformer adapted for time series.",
        "TCN": "Temporal Convolutional Network for sequence modeling.",
        "CNN_1D": "1D convolutional network for time series.",
        "XGBoost_TS": "XGBoost applied to time series features.",
        "RandomForest_TS": "Random Forest for time series classification.",
    },
}

# json3: XAI usage in industries (domain-to-method mapping for bonus logic)
JSON3 = {
    "xai_usage_in_industries": {
        "healthcare": {"SHAP": 1.2, "LIME": 0.9, "PFI": 0.9, "PDP": 0.9},
        "health and medicine": {"SHAP": 1.2, "LIME": 0.9, "PFI": 0.9, "PDP": 0.9},
        "health": {"SHAP": 1.2, "LIME": 0.9, "PFI": 0.9, "PDP": 0.9},
        "finance": {"SHAP": 0.95, "LIME": 0.95, "PFI": 0.95, "PDP": 1.1},
        "financial": {"SHAP": 0.95, "LIME": 0.95, "PFI": 0.95, "PDP": 1.1},
        "manufacturing": {"SHAP": 0.92, "LIME": 1.15, "PFI": 0.92, "PDP": 0.92},
        "industry": {"SHAP": 0.92, "LIME": 1.15, "PFI": 0.92, "PDP": 0.92},
        "cybersecurity": {"SHAP": 0.95, "LIME": 1.1, "PFI": 0.95, "PDP": 0.95},
        "autonomous_vehicles": {"SHAP": 1.1, "LIME": 0.95, "PFI": 0.95, "PDP": 0.95},
        "autonomous vehicles": {"SHAP": 1.1, "LIME": 0.95, "PFI": 0.95, "PDP": 0.95},
        "recommendation_systems": {"SHAP": 0.98, "LIME": 0.98, "PFI": 0.98, "PDP": 1.05},
        "recommendation systems": {"SHAP": 0.98, "LIME": 0.98, "PFI": 0.98, "PDP": 1.05},
        "iot": {"SHAP": 0.97, "LIME": 0.97, "PFI": 1.08, "PDP": 0.97},
        "iot / sensors": {"SHAP": 0.97, "LIME": 0.97, "PFI": 1.08, "PDP": 0.97},
        "general": {"SHAP": 1.0, "LIME": 1.0, "PFI": 1.0, "PDP": 1.0},
    },
}


def main():
    os.makedirs(SURVEY_DIR, exist_ok=True)
    for name, data in [("json1", JSON1), ("json2", JSON2), ("json3", JSON3)]:
        path = os.path.join(SURVEY_DIR, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Wrote {path}")
    print(f"\nDefault survey JSONs created in {SURVEY_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
