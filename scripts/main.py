# main_pipeline.py
import time

# --- Import core classes ---
from utils.temperature_monitor import TemperatureMonitor    
from data_processor import DatasetProcessor                 
from model_manager import ModelManager                     


# --- Import visualization classes  ---
from visualization.metrics_visualizer import MetricsVisualizer
from visualization.fiftyone_visualizer import FiftyOneVisualizer

# --- Import utility modules ---
from utils.config_logging import *
from utils.project_config import *


if __name__ == "__main__":
    
    # --- Initialize logging system ---
    setup_logging()

    # --- Initialize core processing modules ---
    # DatasetProcessor handles preprocessing, augmentation, and dataset splitting
    # ModelManager handles training, evaluation, and post-training analysis
    data_processor = DatasetProcessor(PATHS, DATA_PROCESSOR_CONFIG, YOLO_AUGMENTER_CONFIG)
    model_manager = ModelManager(PATHS, MODEL_MANAGER_CONFIG, MODEL_TRAINING_CONFIG )
    
    start_time: float = time.time()
    logger.info("==============================================")
    logger.info("Starting YOLO Detection Pipeline")
    logger.info("==============================================")

    # --- STAGE 1: Dataset preprocessing and preparation ---
    # Converts COCO annotations to YOLO format, performs stratified split,
    # and optionally applies data augmentation on training images
    # data_processor.run_preprocessing_pipeline(augment=True)
    logger.info("--- STAGE 1: PREPROCESSING COMPLETED ---")

    # --- UTILITY: Visualization of validation set using FiftyOne ---
    # Inspect the validation dataset with FiftyOne to ensure data integrity
    # visualizer = FiftyOneVisualizer(DATASET_DIR, "yolo", "val")
    # visualizer.visualize()
    
    # --- STAGE 2: Model training with temperature monitoring ---
    # Wrap training with TemperatureMonitor to prevent overheating of CPU/GPU
    monitor = TemperatureMonitor(
        func= model_manager.train_multiple_models,
        gpu_temp_threshold=67,
        cpu_temp_threshold=95,
        monitor_interval=30,
        max_consecutive_warnings=3,
    )
    # monitor.start()
    logger.info("--- STAGE 2: TRAINING COMPLETED ---")

    # --- STAGE 3: Model evaluation and post-training analysis ---
    # Evaluate all trained models on the validation split
    # Perform clustering analysis and video prediction/tracking on a specific model
    # model_manager.evaluate_all_models(split='val')
    # model_manager.run_post_training_analysis(model_name="finetuning") 
    logger.info("--- STAGE 3: EVALUATION COMPLETED ---")

    # --- UTILITY: Plot YOLO training metrics ---
    # Visualize training and validation metrics from YOLO CSV logs
    visualizer = MetricsVisualizer(MODEL_TRAINING_CONFIG["finetuning"]["yolo_csv"])
    visualizer.plot_yolo_metrics()

    # --- Pipeline completion and timing ---
    # Log total execution time for performance tracking
    end_time: float = time.time() 
    elapsed: float = end_time - start_time
    logger.info("==============================================")
    logger.info("Pipeline Finished.")
    logger.info(f"Total execution time: {elapsed:.4f} seconds")
    logger.info("==============================================")