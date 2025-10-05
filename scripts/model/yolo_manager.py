# yolo_manager.py
import numpy as np
import torch
import mlflow
import mlflow.bedrock
import tensorboard
from ultralytics import YOLO, settings
from ultralytics.utils.benchmarks import benchmark
from pathlib import Path
from typing import Dict, Any, Union
from utils.config_logging import logger


class YOLOManager:
    """
    Manager class for Ultralytics YOLO models, with MLflow and TensorBoard integration.
    """

    def __init__(self, mlflow_uri: Path):
        """
        Initializes the YOLOManager, sets up MLflow tracking, and enables TensorBoard.
        
        Args:
            mlflow_uri: The URI for MLflow tracking. Defaults to the one in your script.
        """

        # Set MLflow tracking URI 
        mlflow.set_tracking_uri(mlflow_uri)
        logger.info(f"MLflow tracking set to: {mlflow.get_tracking_uri()}")

        # Disable Bedrock auto-tracing since we are training locally and not using AWS Bedrock
        mlflow.bedrock.autolog(disable=True)   
        
        settings.update({"tensorboard": True, "mlflow" : True})
        logger.info("TensorBoard logging enabled.")
        
        # Determine device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")


    def train_model(
        self, 
        model_file: str, 
        dataset_yaml: str, 
        cfg_yaml: str, 
        project_dir: str, 
        name: str, 
        batch_size: int = 32,
        epochs: int = 200
    ) -> YOLO:
        """
        Loads a YOLO model, applies layer freezing based on the unfreeze level, and starts training.
        
        Args:
            model_file: Path to the pre-trained weights (e.g., 'yolov8n.pt').
            dataset_yaml: Path to the dataset YAML file.
            cfg_yaml: Path to the model configuration file (e.g., 'yolov8n.yaml').
            project_dir: Directory to save results to.
            name: Experiment run name.
            batch_size: Training batch size.
            epochs: Number of training epochs.
            
        Returns:
            YOLO: The trained YOLO model object.
        """
        logger.info(f"--- Starting Training Run: {name} ---")
        model: YOLO = YOLO(model_file)
        
        # Start Training
        model.train(
            data=dataset_yaml,
            cfg=cfg_yaml,
            project=project_dir,
            name=name,
            batch=batch_size,
            device=self.device,
            epochs=epochs,
            # Add other common training arguments here if needed (e.g., patience, imgsz)
        )
        logger.info("--- Training Completed ---")
        return model


    def evaluate_model(self, model_file: str, dataset_yaml: str, val_results_dir: str, split='val') -> Dict[str, Any]:
        """
        Evaluates a trained YOLO model on a specified validation set and extracts key metrics.

        Args:
            model_file: Path to the trained model weights (e.g., 'runs/train/exp/weights/best.pt').
            dataset_yaml: Path to the dataset YAML file containing the validation paths.
            val_results_dir: Directory where evaluation results (e.g., confusion matrix) will be saved.
            split: Dataset split to evaluate ('val' by default).

        Returns:
            Dict[str, Any]: A dictionary containing the evaluation metrics, including:
                - 'map50-95': mean Average Precision (mAP) over IoU 0.5:0.95
                - 'map50': mAP at IoU 0.5
                - 'map75': mAP at IoU 0.75
                - 'precision': model precision
                - 'recall': model recall
                - 'f1_score': F1-score calculated from precision and recall
                - 'confusion_matrix_path': Path to the saved confusion matrix image
        """

        logger.info(f"--- Starting Evaluation of Model: {model_file} ---")
        model: YOLO = YOLO(model_file)

        # Run model validation (evaluation)
        metrics: Any = model.val(data=dataset_yaml, split=split)
        
        # Extract metrics into a standard dictionary
         # Includes mAP values, precision, recall, F1-score, and path to confusion matrix
        results: Dict[str, Any] = {
            'map50-95': metrics.box.map,
            'map50': metrics.box.map50,
            'map75': metrics.box.map75,
            'precision': metrics.box.p,
            'recall': metrics.box.r,
            'confusion_matrix_path': str(Path(metrics.save_dir) / 'confusion_matrix.png'),
            'f1_score': np.mean(np.where((metrics.box.p + metrics.box.r) > 0, 
                                        2 * metrics.box.p * metrics.box.r / (metrics.box.p + metrics.box.r), 0))
        }

        
        logger.info("Evaluation Metrics:")
        logger.info(results)
        logger.info("--- Evaluation Completed ---")
        
        return results


    def run_prediction(self, model_file: str, source: Union[str, Path], conf: float = 0.5, iou: float = 0.7):
        """
        Runs prediction/inference on a given image or video source in streaming mode.
        
        Args:
            model_file: Path to the trained model weights.
            source: Path to the image file, video file, or directory.
            conf: Confidence threshold for predictions.
            iou: IoU threshold for Non-Maximum Suppression (NMS).
        """
        logger.info(f"--- Running Prediction on {source} ---")
        model: YOLO = YOLO(model_file)

        # Stream frame by frame to avoid RAM accumulation
        for _ in model.predict(
            source, 
            stream=True, 
            show=True,  # Display results in real-time
            save=True,  # Saves output images/videos
            conf=conf, 
            iou=iou,
            device=self.device
        ):
            pass  #< Process each frame; nothing is stored in RAM

        logger.info(f"Prediction results saved to 'runs/detect/predict...' directory.")


    def run_tracking(self, model_file: str, source: Union[str, Path], conf: float = 0.25, iou: float = 0.7):
        """
        Runs object tracking on a video source in streaming mode.
        
        Args:
            model_file: Path to the trained model weights.
            source: Path to the video file or stream URL.
            conf: Confidence threshold for detections.
            iou: IoU threshold for tracking NMS.
        """
        logger.info(f"--- Running Tracking on {source} ---")
        model: YOLO = YOLO(model_file)

         # Stream frame by frame to avoid RAM accumulation
        for _ in model.track(
            source, 
            stream=True, 
            show=True,  # Display results in real-time
            save=True,  # Saves output video with tracks
            conf=conf, 
            iou=iou,
            device=self.device
        ):
            pass  #< Process each frame; nothing is stored in RAM

        logger.info("Tracking results saved to 'runs/track/track...' directory.")


    @staticmethod
    def run_benchmark(model: str, data: str, imgsz: int = 640, half: bool = False, device: int = 0) -> None:
        """
        Runs the standard Ultralytics benchmark on a model.
        
        Args:
            model: Model name or path (e.g., 'yolov8n.pt').
            data: Dataset YAML (e.g., 'coco128.yaml').
            imgsz: Image size for benchmarking.
            half: Use half-precision (FP16).
            device: GPU device ID (0 for first GPU, -1 for CPU).
        """
        logger.info(f"--- Running Benchmark for {model} ---")
        benchmark(model=model, data=data, imgsz=imgsz, half=half, device=device)
        logger.info("--- Benchmark Completed ---")