#!/usr/bin/env python3
"""
Dataset Metadata Collection Script
Collects comprehensive metadata for real benchmark datasets (image, text, time series)
"""

import json
import csv
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class DatasetMetadata:
    """Structure for dataset metadata"""
    dataset_id: str
    dataset_name: str
    data_type: str  # image, text, timeseries, tabular
    domain: str
    size: int
    type: str  # real, synthetic
    dataset_task: str  # classification, detection, etc.
    description: str
    
    # Image-specific fields
    image_width: int = None
    image_height: int = None
    channels: int = None
    
    # Text-specific fields
    avg_doc_length: int = None
    vocab_size: int = None
    max_length: int = None
    
    # Time series-specific fields
    series_length: int = None
    num_channels: int = None
    sampling_rate: float = None
    
    # Common fields
    num_classes: int = None
    feature_count: int = None
    numeric_features: int = None
    cat_features: int = None
    NaN_values: int = 0
    
    # Additional metadata
    has_masks: bool = False
    has_rationales: bool = False
    has_bboxes: bool = False
    ground_truth_type: str = None
    download_link: str = None
    citation: str = None
    license: str = None
    source: str = None
    paper_reference: str = None
    year_published: int = None
    task_type: str = None

# Image Datasets - General Computer Vision Benchmarks
IMAGE_DATASETS_GENERAL = [
    {
        "dataset_id": "img_001",
        "dataset_name": "ImageNet (ILSVRC 2012)",
        "data_type": "image",
        "domain": "general",
        "size": 1200000,
        "type": "real",
        "dataset_task": "classification",
        "description": "Large-scale image classification dataset with 1.2M images and 1000 classes",
        "image_width": 224,
        "image_height": 224,
        "channels": 3,
        "num_classes": 1000,
        "task_type": "classification",
        "download_link": "https://www.image-net.org/",
        "citation": "Deng, J., et al. 'ImageNet: A large-scale hierarchical image database.' CVPR 2009.",
        "license": "Custom (research use)",
        "source": "ImageNet website",
        "paper_reference": "ImageNet: A large-scale hierarchical image database. CVPR 2009",
        "year_published": 2009,
        "has_masks": False,
        "has_bboxes": False
    },
    {
        "dataset_id": "img_002",
        "dataset_name": "CIFAR-10",
        "data_type": "image",
        "domain": "general",
        "size": 60000,
        "type": "real",
        "dataset_task": "classification",
        "description": "60K color images in 10 classes, 32x32 pixels",
        "image_width": 32,
        "image_height": 32,
        "channels": 3,
        "num_classes": 10,
        "task_type": "classification",
        "download_link": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "citation": "Krizhevsky, A. 'Learning multiple layers of features from tiny images.' 2009.",
        "license": "MIT",
        "source": "University of Toronto",
        "paper_reference": "Learning multiple layers of features from tiny images. 2009",
        "year_published": 2009,
        "has_masks": False,
        "has_bboxes": False
    },
    {
        "dataset_id": "img_003",
        "dataset_name": "CIFAR-100",
        "data_type": "image",
        "domain": "general",
        "size": 60000,
        "type": "real",
        "dataset_task": "classification",
        "description": "60K color images in 100 classes, 32x32 pixels",
        "image_width": 32,
        "image_height": 32,
        "channels": 3,
        "num_classes": 100,
        "task_type": "classification",
        "download_link": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "citation": "Krizhevsky, A. 'Learning multiple layers of features from tiny images.' 2009.",
        "license": "MIT",
        "source": "University of Toronto",
        "paper_reference": "Learning multiple layers of features from tiny images. 2009",
        "year_published": 2009,
        "has_masks": False,
        "has_bboxes": False
    },
    {
        "dataset_id": "img_004",
        "dataset_name": "MNIST",
        "data_type": "image",
        "domain": "general",
        "size": 70000,
        "type": "real",
        "dataset_task": "classification",
        "description": "70K grayscale images of handwritten digits, 28x28 pixels",
        "image_width": 28,
        "image_height": 28,
        "channels": 1,
        "num_classes": 10,
        "task_type": "classification",
        "download_link": "http://yann.lecun.com/exdb/mnist/",
        "citation": "LeCun, Y., et al. 'Gradient-based learning applied to document recognition.' 1998.",
        "license": "Custom",
        "source": "Yann LeCun's website",
        "paper_reference": "Gradient-based learning applied to document recognition. 1998",
        "year_published": 1998,
        "has_masks": False,
        "has_bboxes": False
    },
    {
        "dataset_id": "img_005",
        "dataset_name": "Fashion-MNIST",
        "data_type": "image",
        "domain": "general",
        "size": 70000,
        "type": "real",
        "dataset_task": "classification",
        "description": "70K grayscale images of fashion items, 28x28 pixels, 10 classes",
        "image_width": 28,
        "image_height": 28,
        "channels": 1,
        "num_classes": 10,
        "task_type": "classification",
        "download_link": "https://github.com/zalandoresearch/fashion-mnist",
        "citation": "Xiao, H., et al. 'Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.' 2017.",
        "license": "MIT",
        "source": "Zalando Research",
        "paper_reference": "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. 2017",
        "year_published": 2017,
        "has_masks": False,
        "has_bboxes": False
    },
    {
        "dataset_id": "img_006",
        "dataset_name": "SVHN",
        "data_type": "image",
        "domain": "general",
        "size": 600000,
        "type": "real",
        "dataset_task": "classification",
        "description": "Street View House Numbers dataset, 600K+ images, 32x32 pixels",
        "image_width": 32,
        "image_height": 32,
        "channels": 3,
        "num_classes": 10,
        "task_type": "classification",
        "download_link": "http://ufldl.stanford.edu/housenumbers/",
        "citation": "Netzer, Y., et al. 'Reading digits in natural images with unsupervised feature learning.' NIPS 2011.",
        "license": "Custom",
        "source": "Stanford University",
        "paper_reference": "Reading digits in natural images with unsupervised feature learning. NIPS 2011",
        "year_published": 2011,
        "has_masks": False,
        "has_bboxes": True
    },
    {
        "dataset_id": "img_007",
        "dataset_name": "STL-10",
        "data_type": "image",
        "domain": "general",
        "size": 13000,
        "type": "real",
        "dataset_task": "classification",
        "description": "13K images in 10 classes, 96x96 pixels, designed for unsupervised learning",
        "image_width": 96,
        "image_height": 96,
        "channels": 3,
        "num_classes": 10,
        "task_type": "classification",
        "download_link": "https://cs.stanford.edu/~acoates/stl10/",
        "citation": "Coates, A., et al. 'An analysis of single-layer networks in unsupervised feature learning.' AISTATS 2011.",
        "license": "Custom",
        "source": "Stanford University",
        "paper_reference": "An analysis of single-layer networks in unsupervised feature learning. AISTATS 2011",
        "year_published": 2011,
        "has_masks": False,
        "has_bboxes": False
    },
    {
        "dataset_id": "img_008",
        "dataset_name": "Caltech-101",
        "data_type": "image",
        "domain": "general",
        "size": 9000,
        "type": "real",
        "dataset_task": "classification",
        "description": "9K images in 101 object categories",
        "image_width": 300,
        "image_height": 300,
        "channels": 3,
        "num_classes": 101,
        "task_type": "classification",
        "download_link": "http://www.vision.caltech.edu/Image_Datasets/Caltech101/",
        "citation": "Fei-Fei, L., et al. 'Learning generative visual models from few training examples.' CVPR 2004.",
        "license": "Custom",
        "source": "Caltech",
        "paper_reference": "Learning generative visual models from few training examples. CVPR 2004",
        "year_published": 2004,
        "has_masks": False,
        "has_bboxes": False
    },
    {
        "dataset_id": "img_009",
        "dataset_name": "Caltech-256",
        "data_type": "image",
        "domain": "general",
        "size": 30000,
        "type": "real",
        "dataset_task": "classification",
        "description": "30K images in 256 object categories",
        "image_width": 300,
        "image_height": 300,
        "channels": 3,
        "num_classes": 256,
        "task_type": "classification",
        "download_link": "http://www.vision.caltech.edu/Image_Datasets/Caltech256/",
        "citation": "Griffin, G., et al. 'Caltech-256 object category dataset.' 2007.",
        "license": "Custom",
        "source": "Caltech",
        "paper_reference": "Caltech-256 object category dataset. 2007",
        "year_published": 2007,
        "has_masks": False,
        "has_bboxes": False
    },
    {
        "dataset_id": "img_010",
        "dataset_name": "Places365",
        "data_type": "image",
        "domain": "general",
        "size": 1800000,
        "type": "real",
        "dataset_task": "classification",
        "description": "1.8M images in 365 scene categories for scene recognition",
        "image_width": 256,
        "image_height": 256,
        "channels": 3,
        "num_classes": 365,
        "task_type": "classification",
        "download_link": "http://places2.csail.mit.edu/",
        "citation": "Zhou, B., et al. 'Places: A 10 million Image Database for Scene Recognition.' TPAMI 2017.",
        "license": "Custom",
        "source": "MIT",
        "paper_reference": "Places: A 10 million Image Database for Scene Recognition. TPAMI 2017",
        "year_published": 2017,
        "has_masks": False,
        "has_bboxes": False
    },
    {
        "dataset_id": "img_011",
        "dataset_name": "COCO",
        "data_type": "image",
        "domain": "general",
        "size": 330000,
        "type": "real",
        "dataset_task": "detection",
        "description": "Common Objects in Context, 330K images with object detection and segmentation annotations",
        "image_width": 640,
        "image_height": 480,
        "channels": 3,
        "num_classes": 80,
        "task_type": "detection",
        "download_link": "https://cocodataset.org/",
        "citation": "Lin, T., et al. 'Microsoft COCO: Common Objects in Context.' ECCV 2014.",
        "license": "CC BY 4.0",
        "source": "COCO Consortium",
        "paper_reference": "Microsoft COCO: Common Objects in Context. ECCV 2014",
        "year_published": 2014,
        "has_masks": True,
        "has_bboxes": True
    },
    {
        "dataset_id": "img_012",
        "dataset_name": "Pascal VOC",
        "data_type": "image",
        "domain": "general",
        "size": 20000,
        "type": "real",
        "dataset_task": "detection",
        "description": "Pascal Visual Object Classes, 20K images with 20 object classes",
        "image_width": 500,
        "image_height": 375,
        "channels": 3,
        "num_classes": 20,
        "task_type": "detection",
        "download_link": "http://host.robots.ox.ac.uk/pascal/VOC/",
        "citation": "Everingham, M., et al. 'The Pascal Visual Object Classes (VOC) Challenge.' IJCV 2010.",
        "license": "Custom",
        "source": "PASCAL Challenge",
        "paper_reference": "The Pascal Visual Object Classes (VOC) Challenge. IJCV 2010",
        "year_published": 2010,
        "has_masks": True,
        "has_bboxes": True
    },
    {
        "dataset_id": "img_013",
        "dataset_name": "Open Images",
        "data_type": "image",
        "domain": "general",
        "size": 9000000,
        "type": "real",
        "dataset_task": "detection",
        "description": "9M images with 600+ object classes, largest image dataset",
        "image_width": 1024,
        "image_height": 768,
        "channels": 3,
        "num_classes": 600,
        "task_type": "detection",
        "download_link": "https://storage.googleapis.com/openimages/web/index.html",
        "citation": "Krasin, I., et al. 'OpenImages: A public dataset for large-scale multi-label and multi-class image classification.' 2017.",
        "license": "CC BY 4.0",
        "source": "Google",
        "paper_reference": "OpenImages: A public dataset for large-scale multi-label and multi-class image classification. 2017",
        "year_published": 2017,
        "has_masks": True,
        "has_bboxes": True
    },
    {
        "dataset_id": "img_014",
        "dataset_name": "ImageNet-21K",
        "data_type": "image",
        "domain": "general",
        "size": 14000000,
        "type": "real",
        "dataset_task": "classification",
        "description": "14M images in 21K classes, extended version of ImageNet",
        "image_width": 224,
        "image_height": 224,
        "channels": 3,
        "num_classes": 21000,
        "task_type": "classification",
        "download_link": "https://www.image-net.org/",
        "citation": "Deng, J., et al. 'ImageNet: A large-scale hierarchical image database.' CVPR 2009.",
        "license": "Custom (research use)",
        "source": "ImageNet website",
        "paper_reference": "ImageNet: A large-scale hierarchical image database. CVPR 2009",
        "year_published": 2009,
        "has_masks": False,
        "has_bboxes": False
    },
    {
        "dataset_id": "img_015",
        "dataset_name": "Tiny ImageNet",
        "data_type": "image",
        "domain": "general",
        "size": 100000,
        "type": "real",
        "dataset_task": "classification",
        "description": "100K images in 200 classes, 64x64 pixels, subset of ImageNet",
        "image_width": 64,
        "image_height": 64,
        "channels": 3,
        "num_classes": 200,
        "task_type": "classification",
        "download_link": "http://cs231n.stanford.edu/projects/",
        "citation": "Stanford CS231n course project dataset",
        "license": "Custom",
        "source": "Stanford University",
        "paper_reference": "Stanford CS231n course",
        "year_published": 2015,
        "has_masks": False,
        "has_bboxes": False
    }
]

# Continue with more datasets in the next part...
# This is a comprehensive list that will be expanded

def save_datasets_to_json(datasets: List[Dict], filename: str):
    """Save dataset metadata to JSON file"""
    with open(filename, 'w') as f:
        json.dump(datasets, f, indent=2)

def save_datasets_to_csv(datasets: List[Dict], filename: str):
    """Save dataset metadata to CSV file"""
    if not datasets:
        return
    
    fieldnames = list(datasets[0].keys())
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(datasets)

if __name__ == "__main__":
    # For now, save the initial image datasets
    all_datasets = IMAGE_DATASETS_GENERAL.copy()
    
    # Save to JSON and CSV
    save_datasets_to_json(all_datasets, "image_datasets_metadata.json")
    save_datasets_to_csv(all_datasets, "image_datasets_metadata.csv")
    
    print(f"Saved {len(all_datasets)} image datasets to metadata files")
