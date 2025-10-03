# project_config.py
from pathlib import Path
from typing import Dict, Union


# ================================
# BASE PROJECT PATHS
# ================================
MAIN_PATH = Path(__file__).parent.parent.parent
YAMLS_PATH = MAIN_PATH / 'yamls'
DATASET_PATH = MAIN_PATH / 'dataset'
LABELS_PATH = DATASET_PATH / 'labels'
IMAGES_PATH = DATASET_PATH / 'images'

# COCO Annotations
ORIGINAL_COCO_JSON_FILE = LABELS_PATH / 'original_coco_labels.json'
COCO_JSON_FILE = LABELS_PATH / 'coco_labels.json'

# YOLO Configuration
YOLO_DATASET_PATH = DATASET_PATH / 'yolo_dataset.yaml'      # For augmented data
YOLO_DATASET_PATH2 = DATASET_PATH / 'yolo_dataset2.yaml'    # For base data / transfer learning

# Images and Mapping
ORIGINAL_NAMES_MAP = IMAGES_PATH / 'original_names_map.json'
UNPROCESSED_IMAGES_PATH = IMAGES_PATH / 'unprocessed'
RENAMED_IMAGES_PATH = IMAGES_PATH / 'renamed'
RENAMED_LABELS_PATH = LABELS_PATH / 'renamed'

# Splits and Augmentation
VAL_IMAGES_PATH = IMAGES_PATH / 'val'
TRAIN_IMAGES_PATH = IMAGES_PATH / 'train'
TRAIN_LABELS_PATH = IMAGES_PATH / 'train'
TRAIN_AUG_IMAGES_PATH = IMAGES_PATH / 'train_augmented'
TRAIN_AUG_LABELS_PATH = LABELS_PATH / 'train_augmented'

# Model Results
TRAIN_RESULTS_PATH = MAIN_PATH / 'training_results'
VAL_RESULTS_PATH = MAIN_PATH / 'val_results'
MLFLOW_PATH = MAIN_PATH / "mlflow"

# Videos
VIDEO_SOURCE = MAIN_PATH / 'videos' / "LA.mp4"


# ================================
# CENTRALIZED PATHS
# ================================
PATHS: Dict[str, Path] = {
    # Base project paths
    "main_path": MAIN_PATH,
    "yamls_path": YAMLS_PATH,
    "dataset_path": DATASET_PATH,
    "labels_path": LABELS_PATH,
    "images_path": IMAGES_PATH,

    # COCO Annotations
    "original_coco_json_file": ORIGINAL_COCO_JSON_FILE,
    "coco_json_file": COCO_JSON_FILE,

    # YOLO Configuration
    "yolo_dataset_path": YOLO_DATASET_PATH,
    "yolo_dataset_path2": YOLO_DATASET_PATH2,

    # Images and Mapping
    "original_names_map": ORIGINAL_NAMES_MAP,
    "unprocessed_images_path": UNPROCESSED_IMAGES_PATH,
    "renamed_images_path": RENAMED_IMAGES_PATH,
    "renamed_labels_path": RENAMED_LABELS_PATH,

    # Splits and Augmentation
    "val_images_path": VAL_IMAGES_PATH,
    "train_images_path": TRAIN_IMAGES_PATH,
    "train_labels_path": TRAIN_LABELS_PATH,
    "train_aug_images_path": TRAIN_AUG_IMAGES_PATH,
    "train_aug_labels_path": TRAIN_AUG_LABELS_PATH,

    # Model Results
    "train_results_path": TRAIN_RESULTS_PATH,
    "val_results_path": VAL_RESULTS_PATH,
    "mlflow_path": MLFLOW_PATH,

    # Videos
    "video_source": VIDEO_SOURCE,
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
    "model_from_scratch": 
    {
        "weights": YAMLS_PATH / 'yolo11n_example.yaml', # YOLO model structure config
        "config_yml": YAMLS_PATH / "sgd_from_scratch.yaml",
        "data_yml": PATHS["yolo_dataset_path"],
    },

    # 2. Transfer Learning (Pre-trained YOLO11n, Freeze Backbone) - AdamW
    "model_transfer_learning": 
    {
        "weights": 'yolo11n.pt',
        "config_yml": YAMLS_PATH / "adamw_transfer_learning.yaml",
        "data_yml": PATHS["yolo_dataset_path2"],
    },
    
    # 3. Fine-Tuning (Pre-trained YOLO11n, Train All) - AdamW
    "model_finetuning": 
    {
        "weights": 'yolo11n.pt',
        "config_yml": YAMLS_PATH / "adamw_finetuning.yaml",
        "data_yml": PATHS["yolo_dataset_path2"],
    },
}