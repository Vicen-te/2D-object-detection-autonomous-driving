# main_pipeline.py
import time
from pathlib import Path
from typing import Dict, Any

from utils.temperature_monitor import TemperatureMonitor
from data_processor import DatasetProcessor
from model_manager import ModelManager

# Utility functions
from visualization.fiftyone_visualizer import FiftyOneVisualizer

from utils.config_logging import *


class PipelineRunner:
    """
    Main class that orchestrates the CV/ML workflow, defining paths and 
    configurations, and executing all stages.
    """

    def __init__(self, main_path: Path):
        """
        Defines and consolidates all project paths and configurations.
        """
        self.main_path = main_path
        self.paths: Dict[str, Path] = self._define_paths()
        self.config: Dict[str, Any] = self._define_config()
        
        # Initialize business logic modules
        self.data_processor = DatasetProcessor(self.paths)
        self.model_manager = ModelManager(self.paths, self.config)

    def _define_paths(self) -> Dict[str, Path]:
        """ Defines all project paths in a centralized dictionary. """
        
        # Base Paths
        dataset_path: Path = self.main_path / 'dataset'
        yamls_path: Path = self.main_path / "yamls"
        labels_path: Path = dataset_path / 'labels'
        images_path: Path = dataset_path / 'images'

        paths = {
            # Base
            'main_path': self.main_path,
            'dataset_path': dataset_path,
            'yamls_path': yamls_path,
            'labels_path': labels_path,
            'images_path': images_path,
            
            # COCO Annotations
            'original_coco_json_file': labels_path / 'original_coco_labels.json',
            'coco_json_file': labels_path / 'coco_labels.json',
            
            # YOLO Configuration
            'yolo_dataset_path': dataset_path / 'yolo_dataset.yaml',      #< For augmented data
            'yolo_dataset_path2': dataset_path / 'yolo_dataset2.yaml',    #< For base data/transfer learning
            
            # Images and Mapping
            'original_names_map': images_path / 'original_names_map.json',
            'unprocessed_images_path': images_path / 'unprocessed',
            'renamed_images_path': images_path / 'renamed',
            'renamed_labels_path': labels_path / 'renamed', 

            # Splits and Augmentation
            'train_aug_images_path': images_path / 'train_augmented',
            'train_aug_labels_path': labels_path / 'train_augmented',
            
            # Model Results
            'train_results_path': self.main_path / 'training_results',
            'val_results_path': self.main_path / 'val_results',
            'mlflow': self.main_path / "mlflow",
            
            # Videos (For Prediction/Tracking Demo)
            'video_source': self.main_path / 'videos' / "LA.mp4",
        }
        return paths

    def _define_config(self) -> Dict[str, Any]:
        """ Defines all training and process configurations. """
        return {
            'yolo_base_model': 'yolo11n.pt', # Base model for Fine-tuning/Transfer Learning
            'augmentations_per_image': 3,
            'max_workers': 8,
            'num_clusters': 7,
            'random_state': 100107,
            'conf_thres': 0.5,
            'iou_thres': 0.4,
            'unfreeze_layers': 2 
        }

    def run_full_pipeline(self) -> None:
        """ Executes the entire workflow (Dataset -> Train -> Analyze). """
        
        start_time: float = time.time()
        logger.info("==============================================")
        logger.info("Starting YOLO Detection Pipeline")
        logger.info("==============================================")

        # --- STAGE 1: DATASET PREPROCESSING AND PREPARATION ---
        # self.processor.run_preprocessing_pipeline(augment=True)
        logger.info("--- STAGE 1: PREPROCESSING COMPLETED ---")
        
        # --- UTILITY: VISUALIZATION ---
        # visualizer = FiftyOneVisualizer(self.paths['dataset_path'], "yolo", "val")
        #visualizer.visualize()
        
        # --- STAGE 2: MODEL TRAINING ---
        monitor = TemperatureMonitor(
            func= self.model_manager.train_multiple_models,
            gpu_temp_threshold=67,
            cpu_temp_threshold=95,
            monitor_interval=30,
            max_consecutive_warnings=3,
        )
        # monitor.start()
        logger.info("--- STAGE 2: TRAINING COMPLETED ---")

        # --- STAGE 3: EVALUATION AND POST-TRAINING ANALYSIS ---
        # self.trainer.evaluate_all_models(split='val')
        self.model_manager.run_post_training_analysis(model_name='model_finetuning') 
        logger.info("--- STAGE 3: EVALUATION COMPLETED ---")

        end_time: float = time.time() 
        elapsed: float = end_time - start_time
        logger.info("==============================================")
        logger.info("Pipeline Finished.")
        logger.info(f"Total execution time: {elapsed:.4f} seconds")
        logger.info("==============================================")


def hello(msg: str = "hello"):
    print(msg)
    

if __name__ == "__main__":
    # The main path (the directory containing 'dataset', 'yamls', etc.)
    # Assumes the script is run from a sub-directory, hence parent.parent
    MAIN_PATH = Path(__file__).parent.parent
    setup_logging()
    
    # Instantiate and run the orchestrator
    pipeline = PipelineRunner(MAIN_PATH)
    pipeline.run_full_pipeline()