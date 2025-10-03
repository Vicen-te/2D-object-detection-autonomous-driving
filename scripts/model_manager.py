# model_manager.py
from pathlib import Path
from typing import Dict, Any

from model.yolo_manager import YOLOManager
from model.clustering_analyzer import ClusteringAnalyzer 
from utils.config_logging import logger


class ModelManager:
    """
    Handles the complete lifecycle of YOLO models: 
    training, evaluation, prediction, and post-training analysis.
    """

    def __init__(self, paths: Dict[str, Path], config: Dict[str, Any], training_config: Dict[str, Any]):
        """
        Initializes the ModelManager with project paths, general config, 
        and training configurations for multiple model variants.
        """
        self.paths = paths
        self.config = config
        self.training_config = training_config
        self.manager = YOLOManager(paths["mlflow_path"])  #< Central YOLO management instance


    def train_multiple_models(self) -> None:
        """
        Executes training for all defined model configurations:
        - From Scratch
        - Transfer Learning
        - Fine-Tuning
        """
        logger.info("Starting Multiple Model Training...")

        # Ensure output directories exist
        self.paths['train_results_path'].mkdir(parents=True, exist_ok=True)
        self.paths['val_results_path'].mkdir(parents=True, exist_ok=True)

        # Iterate through all training configurations
        for model_name, cfg in self.training_config.items():
            logger.info(f"--- Training: {model_name} ---")
            self.manager.train_model(
                cfg['weights'], 
                cfg['data_yml'], 
                cfg['config_yml'], 
                self.paths['train_results_path'], 
                model_name
            )
            logger.info(f"  > Training for '{model_name}' completed.")


    def evaluate_all_models(self, split: str = 'val') -> None:
        """
        Evaluates all trained models on a specified dataset split (e.g., train, val, test).
        """
        logger.info("Starting Model Evaluation...")
        
        for model_name, cfg in self.training_config.items():
            model_path: Path = self.paths['train_results_path'] / model_name / 'weights' / 'best.pt'
            val_results_path: Path = self.paths['val_results_path'] / model_name
            
            if model_path.exists():
                logger.info(f"  > Evaluating {model_name} on {split} set...")
                self.manager.evaluate_model(model_path, self.paths['yolo_dataset_path'], val_results_path, split)
                logger.info(f"  > Evaluation for '{model_name}' completed.")
            else:
                logger.error(f"  > Model not found at {model_path}. Skipping evaluation.")
                
        logger.info("Model Evaluation Finished.")


    def run_clustering_analysis(self, model_name: str) -> None:
        """
        Performs clustering analysis on validation images using the trained model.
        """
        logger.info("Starting Clustering Analysis...")

        model_path: Path = self.paths['train_results_path'] / model_name / 'weights' / 'best.pt'
        if not model_path.exists():
            logger.error(f"  > Analysis model '{model_name}' not found. Aborting clustering.")
            return

        logger.info(f"  > Executing Clustering with K=" + 
                    f"{self.config['num_clusters']} on {self.paths["val_images_path"]}...")
        
        analyzer = ClusteringAnalyzer(
            self.config['num_clusters'], 
            self.config['random_state']
        )

        analyzer.run_analysis(
            model_path, 
            self.paths["val_images_path"], 
            self.config["num_samples"], 
            self.config["dimension"], 
            self.config["layer_index"], 
        )

        logger.info("  > Clustering completed.")


    def run_video_prediction_tracking(self, model_name: str) -> None:
        """
        Performs video prediction and tracking using the trained model.
        """
        logger.info("Starting Video Prediction/Tracking...")

        model_path: Path = self.paths['train_results_path'] / model_name / 'weights' / 'best.pt'
        if not model_path.exists():
            logger.error(f"  > Analysis model '{model_name}' not found. Aborting video prediction/tracking.")
            return

        logger.info(f"  > Running Prediction/Tracking on video " +
                    f"{self.paths['video_source'].name}...")
        
        self.manager.run_tracking(
            model_path, 
            self.paths['video_source'], 
            self.config['conf_thres'], 
            self.config['iou_thres']
        )

        self.manager.run_prediction(
            model_path, 
            self.paths['video_source'], 
            self.config['conf_thres'], 
            self.config['iou_thres']
        )

        logger.info("  > Prediction/Tracking completed.")


    def run_post_training_analysis(self, model_name: str) -> None:
        """
        Performs full post-training analysis for a specific model:
        - Clustering analysis on validation images
        - Video prediction and tracking
        """
        self.run_clustering_analysis(model_name)
        self.run_video_prediction_tracking(model_name)