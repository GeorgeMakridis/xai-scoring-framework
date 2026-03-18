# Data Structure and Metrics

## Per-Type Folders (tabular, image, text, timeseries)

### dataset_metadata.csv
Dataset descriptions and characteristics.
- dataset_id, dataset_name, size, type, dataset_task, feature_count, domain, description
- numeric_features, cat_features, NaN_values
- data_type, modality fields (image_width, image_height, channels, series_length, num_channels, avg_doc_length, vocab_size, max_length, etc.)

### ai_model_performance.csv
AI model benchmark results per dataset.
- dataset_id, ai_model_id, Accuracy, Precision

### xai_quantitative_metrics.csv
Quantitative XAI metrics per method per dataset.
- Dataset ID, XAI Method, Fidelity, Simplicity, Stability, Localization, source

### xai_qualitative_ratings.csv
User ratings (1-5) for XAI methods per dataset.
- dataset_id, interpretability_*, understanding_*, trust_* (per method)

## Shared (data/shared/)

### ai_model_definitions.csv
AI model catalog: properties and task types.

### xai_method_definitions.csv
XAI method catalog: Post-Hoc, Ante-Hoc, Global/Local, etc.

### domain_relevance.csv
domain_from, domain_to, relevance (dataset-to-domain relevance)

### dataset_relevance.csv
dataset_id_a, dataset_id_b, relevance (pairwise dataset relevance)
