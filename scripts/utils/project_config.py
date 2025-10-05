# project_config.py
from pathlib import Path
from typing import Dict, Union


# ================================
# BASE PROJECT PATHS
# ================================
MAIN_DIR                = Path(__file__).parent.parent.parent
YAMLS_DIR               = MAIN_DIR / 'yamls'                    # YAML configs: hyperparams, training setups
DATASET_DIR             = MAIN_DIR / 'dataset'                  # Main dataset directory
IMAGES_DIR              = DATASET_DIR / 'images'                # All images
LABELS_DIR              = DATASET_DIR / 'labels'                # COCO / YOLO labels

# COCO Annotations
ORIGINAL_COCO_JSON      = LABELS_DIR / '_annotations.coco.json' # Original COCO annotations
COCO_JSON               = LABELS_DIR / 'coco_labels.json'       # Converted COCO labels

# YOLO Configuration Files
YOLO_AUG_DATASET_YAML   = DATASET_DIR / 'yolo_aug_dataset.yaml'     # YOLO augmented data config
YOLO_BASE_DATASET_YAML  = DATASET_DIR / 'yolo_base_dataset.yaml'    # YOLO base data / transfer learning config

# Images and Mapping
ORIGINAL_NAMES_MAP_FILE = IMAGES_DIR / 'original_names_map.json'
UNPROCESSED_IMAGES_DIR  = IMAGES_DIR / 'unprocessed'            # Raw images before renaming
RENAMED_IMAGES_DIR      = IMAGES_DIR / 'renamed'                # Images with standardized names
RENAMED_LABELS_DIR      = LABELS_DIR / 'renamed'                # Labels corresponding to renamed images

# Dataset Splits
VAL_IMAGES_DIR          = IMAGES_DIR / 'val'                    # Validation images
TRAIN_IMAGES_DIR        = IMAGES_DIR / 'train'                  # Training images
TRAIN_LABELS_DIR        = LABELS_DIR / 'train'                  # Labels for training
TRAIN_AUG_IMAGES_DIR    = IMAGES_DIR / 'train_augmented'        # Augmented training images
TRAIN_AUG_LABELS_DIR    = LABELS_DIR / 'train_augmented'        # Labels for augmented training

# Model Results
TRAIN_RESULTS_DIR       = MAIN_DIR / 'training_results'         # Directory for model outputs
VAL_RESULTS_DIR         = MAIN_DIR / 'val_results'              # Optional validation results
MLFLOW_DIR              = MAIN_DIR / "mlflow"                   # Experiment tracking

# Videos
VIDEOS_DIR              = MAIN_DIR / 'videos'                   # Videos for prediction/tracking
VIDEO_FILE              = VIDEOS_DIR / "LA.mp4"

# ================================
# CENTRALIZED PATHS
# ================================
PATHS: Dict[str, Path] = {
    "main_dir": MAIN_DIR,
    "yamls_dir": YAMLS_DIR,
    "dataset_dir": DATASET_DIR,
    "images_dir": IMAGES_DIR,
    "labels_dir": LABELS_DIR,

    "original_coco_json": ORIGINAL_COCO_JSON,
    "coco_json": COCO_JSON,

    "yolo_aug_dataset_yaml": YOLO_AUG_DATASET_YAML,
    "yolo_base_dataset_yaml": YOLO_BASE_DATASET_YAML,

    "original_names_map_file": ORIGINAL_NAMES_MAP_FILE,
    "unprocessed_images_dir": UNPROCESSED_IMAGES_DIR,
    "renamed_images_dir": RENAMED_IMAGES_DIR,
    "renamed_labels_dir": RENAMED_LABELS_DIR,

    "train_images_dir": TRAIN_IMAGES_DIR,
    "val_images_dir": VAL_IMAGES_DIR,
    "train_labels_dir": TRAIN_LABELS_DIR,
    "train_aug_images_dir": TRAIN_AUG_IMAGES_DIR,
    "train_aug_labels_dir": TRAIN_AUG_LABELS_DIR,

    "train_results_dir": TRAIN_RESULTS_DIR,
    "val_results_dir": VAL_RESULTS_DIR,
    "mlflow_dir": MLFLOW_DIR,

    "videos_dir": VIDEOS_DIR,
    "video_file": VIDEO_FILE,
}


# ================================
# DATA PROCESSOR CONFIG
# ================================
DATA_PROCESSOR_CONFIG: Dict[str, int] = {
    'augmentations_per_image': 3,
    'max_workers': 8,
}


# ================================
# YOLO AUGMENTER CONFIG
# ================================
YOLO_AUGMENTER_CONFIG: Dict[str, int | float] = {
    "max_rotation_deg": 30,
    "max_translation_pct": 0.2,  # 20%
    "min_scale": 0.8,
    "max_scale": 1.2,
}


# ================================
# MODEL MANAGER CONFIG
# ================================
MODEL_MANAGER_CONFIG: Dict[str, Union[str, Path, int, float]] = {
    # clustering
    "dimension": 2,
    'num_clusters': 7,
    "num_samples": 5,
    "layer_index": 10,
    'random_state': 100107,

    # yolo
    'conf_thres': 0.5,
    'iou_thres': 0.4,
}


# ================================
# MODEL TRAINING CONFIGS
# ================================
MODEL_TRAINING_CONFIG: dict[str, dict[str, str | Path]] = {
    # 1. From Scratch (Random Weights) - SGD
    "from_scratch": 
    {
        "weights": YAMLS_DIR / 'yolo11n_example.yaml', # YOLO model structure config
        "config_yml": YAMLS_DIR / "sgd_from_scratch.yaml",
        "data_yml": YOLO_AUG_DATASET_YAML,
        "yolo_csv": TRAIN_RESULTS_DIR / "from_scratch" / "results.csv"  
    },

    # 2. Transfer Learning (Pre-trained YOLO11n, Freeze Backbone) - AdamW
    "transfer_learning": 
    {
        "weights": 'yolo11n.pt',
        "config_yml": YAMLS_DIR / "adamw_transfer_learning.yaml",
        "data_yml": YOLO_BASE_DATASET_YAML,
        "yolo_csv": TRAIN_RESULTS_DIR / "transfer_learning" / "results.csv"
    },
    
    # 3. Fine-Tuning (Pre-trained YOLO11n, Train All) - AdamW
    "finetuning": 
    {
        "weights": 'yolo11n.pt',
        "config_yml": YAMLS_DIR / "adamw_finetuning.yaml",
        "data_yml": YOLO_BASE_DATASET_YAML,
        "yolo_csv": TRAIN_RESULTS_DIR / "finetuning" / "results.csv"
    },
}