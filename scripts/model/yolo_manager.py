# yolo_manager.py
import numpy as np
import torch
import mlflow
import mlflow.bedrock
import tensorboard
from ultralytics import YOLO, settings
from ultralytics.utils.benchmarks import benchmark
from pathlib import Path
from typing import Dict, Any, Union, Optional

from scripts.utils.config_logging import setup_logging
logger = setup_logging()


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
        model_path: str, 
        data_yaml_path: str, 
        cfg_yaml_path: str, 
        project_path: str, 
        name: str, 
        batch_size: int = 32,
        epochs: int = 200
    ) -> YOLO:
        """
        Loads a YOLO model, applies layer freezing based on the unfreeze level, and starts training.
        
        Args:
            model_path: Path to the pre-trained weights (e.g., 'yolov8n.pt').
            data_yaml_path: Path to the dataset YAML file.
            cfg_yaml_path: Path to the model configuration file (e.g., 'yolov8n.yaml').
            project_path: Directory to save results to.
            name: Experiment run name.
            unfreeze_level: 0 (no freezing/full fine-tuning), 1 (freeze backbone), 2 (freeze backbone and neck), 3 (freeze all).
            batch_size: Training batch size.
            epochs: Number of training epochs.
            
        Returns:
            YOLO: The trained YOLO model object.
        """
        logger.info(f"\n--- Starting Training Run: {name} ---")
        model: YOLO = YOLO(model_path)
        
        # Start Training
        model.train(
            data=data_yaml_path,
            cfg=cfg_yaml_path,
            project=project_path,
            name=name,
            batch=batch_size,
            device=self.device,
            epochs=epochs,
            # Add other common training arguments here if needed (e.g., patience, imgsz)
        )
        logger.info("--- Training Completed ---")
        return model

    def evaluate_model(self, model_path: str, data_yaml_path: str, split='val') -> Dict[str, Any]:
        """
        Evaluates a trained model on the validation set specified in the data YAML.
        
        Args:
            model_path: Path to the trained model weights (e.g., 'runs/train/exp/weights/best.pt').
            data_yaml_path: Path to the dataset YAML file containing the 'val' path.
            
        Returns:
            Dict[str, Any]: A dictionary containing the evaluation metrics.
        """
        logger.info(f"\n--- Starting Evaluation of Model: {model_path} ---")
        model: YOLO = YOLO(model_path)

        # Run model validation (evaluation)
        metrics: Any = model.val(data=data_yaml_path, split=split)
        
        # Extract metrics into a standard dictionary
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


        # Visualize the model predictions on the validation set
        # model.plot_results(save_dir=val_results_path)  # This will plot the results of the validation set

        # If you want to visualize a specific image, you can use:
        # results = model.predict(source='path/to/your/image.jpg', conf=0.25, save=True)  # Cambia la ruta a tu imagen
        # 'save=True' guarda las imágenes con las cajas predichas en 'runs/detect/predict' por defecto

        return results


    def run_prediction(self, model_path: str, source: Union[str, Path], conf: float = 0.5, iou: float = 0.7) -> Any:
        """
        Runs prediction/inference on a given image or video source.
        
        Args:
            model_path: Path to the trained model weights.
            source: Path to the image file, video file, or directory.
            conf: Confidence threshold for predictions.
            iou: IoU threshold for Non-Maximum Suppression (NMS).
            
        Returns:
            Any: The results object from the YOLO predict operation.
        """
        logger.info(f"\n--- Running Prediction on {source} ---")
        model: YOLO = YOLO(model_path)

        results: Any = model.predict(
            source, 
            save=True,  # Saves output images/videos with bounding boxes
            conf=conf, 
            iou=iou,
            device=self.device
        )
        logger.info("Prediction results saved to 'runs/detect/predict...' folder.")
        return results


    def run_tracking(self, model_path: str, source: Union[str, Path], conf: float = 0.25, iou: float = 0.7) -> Any:
        """
        Runs object tracking on a given video source (requires a video stream or file).
        
        Args:
            model_path: Path to the trained model weights.
            source: Path to the video file or stream URL.
            conf: Confidence threshold for detections.
            iou: IoU threshold for tracking NMS.
            
        Returns:
            Any: The results object from the YOLO track operation.
        """
        logger.info(f"\n--- Running Tracking on {source} ---")
        model: YOLO = YOLO(model_path)

        results: Any = model.track(
            source, 
            show=True, # Display results in real-time
            save=True, # Saves output video with tracks
            conf=conf, 
            iou=iou,
            device=self.device
        )
        logger.info("Tracking results saved to 'runs/track/track...' folder.")
        return results

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
        logger.info(f"\n--- Running Benchmark for {model} ---")
        benchmark(model=model, data=data, imgsz=imgsz, half=half, device=device)
        logger.info("--- Benchmark Completed ---")