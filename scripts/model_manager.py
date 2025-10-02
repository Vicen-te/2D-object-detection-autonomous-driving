# model_trainer.py
from pathlib import Path
from typing import Dict, Any

from model.yolo_manager import YOLOManager
from model.clustering_analyzer import ClusteringAnalyzer 
from utils.config_logging import logger


class ModelManager:
    """
    Manages the training, evaluation, prediction, and post-training tasks 
    for object detection models (YOLO).
    """

    def __init__(self, paths: Dict[str, Path], config: Dict[str, Any]):
        """
        Initializes the model trainer with paths and training configurations.
        """
        self.paths = paths
        self.config = config
        self.manager = YOLOManager(paths["mlflow"])  #< central manager instance


    def train_multiple_models(self) -> None:
        """
        Executes training for different model configurations 
        (From Scratch, Transfer Learning, Fine-Tuning).
        """
        logger.info("Starting Multiple Model Training...")

        self.paths['train_results_path'].mkdir(parents=True, exist_ok=True)
        self.paths['val_results_path'].mkdir(parents=True, exist_ok=True)

        # Path Definitions
        data_yml_path = self.paths.get('yolo_dataset_path2')
        data_yml_path_base = self.paths.get('yolo_dataset_path', data_yml_path)

        yamls_path = self.paths['yamls_path']
        yolo_base_model = self.config['yolo_base_model'] 

        training_configs = [
            # 1. From Scratch (Random Weights) - SGD
            {
                'name': 'model_from_scratch',
                'weights': self.paths['yamls_path'] / 'yolo11s.yaml', # YOLO model structure config
                'config_yml': yamls_path / 'sgd_from_scratch.yaml',
                'data_yml': data_yml_path_base 
            },
            # 2. Transfer Learning (Pre-trained YOLO11n, Freeze Backbone) - AdamW
            {
                'name': 'model_transfer_learning',
                'weights': yolo_base_model,
                'config_yml': yamls_path / 'adamw_transfer_learning.yaml',
                'data_yml': data_yml_path
            },
            # 3. Fine-Tuning (Pre-trained YOLO11n, Train All) - AdamW
            {
                'name': 'model_finetuning',
                'weights': yolo_base_model,
                'config_yml': yamls_path / 'adamw_finetuning.yaml',
                'data_yml': data_yml_path 
            }
        ]

        for cfg in training_configs:
            logger.info(f"--- Training: {cfg['name']} ---")
            self.manager.train_model(
                cfg['weights'], 
                cfg['data_yml'], 
                cfg['config_yml'], 
                self.paths['train_results_path'], 
                cfg['name']
            )
            logger.info(f"  > Training for '{cfg['name']}' completed.")


    def evaluate_all_models(self, split: str = 'val') -> None:
        """
        Evaluates the trained models on the specified dataset split.
        """
        logger.info("Starting Model Evaluation...")
        
        trained_models = ['model_from_scratch', 'model_transfer_learning', 'model_finetuning']
        data_yml_path = self.paths.get('yolo_dataset_path')
        
        for model_name in trained_models:
            model_path: Path = self.paths['train_results_path'] / model_name / 'weights' / 'best.pt'
            val_results_path: Path = self.paths['val_results_path'] / model_name 
            
            if model_path.exists():
                logger.info(f"  > Evaluating {model_name} on {split} set...")
                self.manager.evaluate_model(model_path, data_yml_path, split)
                logger.info(f"  > Evaluation for '{model_name}' completed.")
            else:
                logger.info(f"  > Model not found at {model_path}. Skipping evaluation.")
                
        logger.info("Model Evaluation Finished.")


    def run_clustering_analysis(self, model_name: str = 'model_finetuning') -> None:
        """
        Performs clustering analysis on validation images using the trained model.
        """
        logger.info("Starting Clustering Analysis...")

        model_path: Path = self.paths['train_results_path'] / model_name / 'weights' / 'best.pt'
        if not model_path.exists():
            logger.info(f"  > Analysis model '{model_name}' not found. Aborting clustering.")
            return

        num_clusters = self.config.get('num_clusters', 7)
        random_state = self.config.get('random_state', 100107)
        val_images_path = self.paths['images_path'] / 'val'

        logger.info(f"  > Executing Clustering with K={num_clusters} on {val_images_path}...")
        analyzer = ClusteringAnalyzer(n_clusters=num_clusters, random_state=random_state)
        analyzer.run_analysis(val_images_path, model_path, num_samples=5)
        logger.info("  > Clustering completed.")


    def run_video_prediction_tracking(self, model_name: str = 'model_finetuning') -> None:
        """
        Performs video prediction and tracking using the trained model.
        """
        logger.info("Starting Video Prediction/Tracking...")

        model_path: Path = self.paths['train_results_path'] / model_name / 'weights' / 'best.pt'
        if not model_path.exists():
            logger.info(f"  > Analysis model '{model_name}' not found. Aborting video prediction/tracking.")
            return

        video_source = self.paths['video_source']
        conf_thres = self.config.get('conf_thres', 0.5)
        iou_thres = self.config.get('iou_thres', 0.4)

        logger.info(f"  > Running Prediction/Tracking on video {video_source.name}...")
        self.manager.run_tracking(model_path, video_source, conf_thres, iou_thres)
        self.manager.run_prediction(model_path, video_source, conf_thres, iou_thres)
        logger.info("  > Prediction/Tracking completed.")


    def run_post_training_analysis(self, model_name: str = 'model_finetuning') -> None:
        """
        Performs full post-training analysis: clustering and video prediction/tracking.
        """
        self.run_clustering_analysis(model_name)
        self.run_video_prediction_tracking(model_name)