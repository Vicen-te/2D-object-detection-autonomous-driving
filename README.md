# 2D Object Detection for Autonomous Driving

This project implements a complete pipeline for 2D object detection in autonomous driving environments. It uses YOLO11 with PyTorch for training and inference, while integrating TensorBoard and MLflow for experiment tracking, and FiftyOne for dataset visualization and model evaluation.

---

## Features

- Train YOLO11 models with configurable hyperparameters.
- Real-time object detection and tracking on images and video streams.
- Automatic experiment logging and visualization with TensorBoard.
- Experiment comparison and results management with MLflow.
- Interactive dataset exploration and error analysis using FiftyOne.
- Modular codebase to extend for research or production use cases.

---

## Installation

1. Clone this repository:

```bash
git clone https://github.com/Vicen-te/2D-object-detection-autonomous-driving.git

cd 2D-object-detection-autonomous-driving
```

2. Create a virtual environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate # Windows
source .venv/bin/activate # Linux/macOS
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

4. Required libraries:

- `ultralytics` – YOLO11 models and training utilities.
- `torch` – Deep learning framework.
- `tensorboard` – Training visualization.
- `mlflow` – Experiment tracking and management.
- `fiftyone` – Dataset inspection and model evaluation.

---


## Project Structure
```bash
2D-object-detection-autonomous-driving/
│
├─ data/
│ ├─ augmentation_yolo.py 		# Data augmentation for YOLO
│ ├─ coco_converter.py 			# Convert datasets to COCO format
│ ├─ dataset_splitter.py 		# Split dataset into train/val/test
│ └─ file_system_manager.py 	# Handle dataset and file operations
│
├─ model/
│ ├─ clustering_analyzer.py 	# Optional clustering analysis of features
│ └─ yolo_manager.py 			# YOLO training, inference, and tracking manager
│
├─ utils/
│ ├─ config_logging.py          # Logging configuration
│ ├─ temperature_monitor.py     # Optional CPU/GPU temperature monitor
│ ├─ project_config.py          # Centralized paths and configurations
│ └─ types_aliases.py           # Type hints and custom aliases
│
├─ visualization/
│ ├─ fiftyone_visualizer.py     # Dataset visualization with FiftyOne (GUI)
│ └─ fiftyone_cli_visualizer.py # Dataset visualization with FiftyOne (CLI)
│
├─ main.py 			            # Orchestrates the full pipeline
├─ data_processor.py 			# Handles preprocessing pipeline
└─ model_manager.py 			# Manages models: training and post-training analysis
```

---

Usage

1. Train a model
```bash
python scripts/main.py
```

2. Monitor with TensorBoard
```bash
tensorboard --logdir training_results/
```

3. Launch MLflow UI
```bash
mlflow ui --backend-store-uri mlflow/
```

4. Explore dataset and results with FiftyOne — You can launch the custom CLI visualizer to explore YOLO/COCO datasets:
```bash
python scripts/visualization/fiftyone_cli_visualizer.py \
  --p <path_to_dataset_root> \    # Path to the dataset root folder
  --f <yolo_or_coco> \            # Dataset format: yolo or coco
  --s <train_val_or_test> \       # Dataset split: train, val, or test (optional, default: val)
  --n <path_to_names_json>        # Path to the original names JSON (optional)
```

---

## Notes

The system supports custom datasets with configurable number of classes.

Training results are automatically logged into both TensorBoard and MLflow.

Experiment reproducibility is ensured through configuration YAML files.

Models can be switched easily between YOLO11n, YOLO11s, YOLO11m, YOLO11l, YOLO11x scales.

---

## License

MIT License © 2025 Vicente Brisa Saez
GitHub: https://github.com/Vicen-te