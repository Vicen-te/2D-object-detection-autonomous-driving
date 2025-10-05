# data_processor.py
import json
from pathlib import Path
from typing import Dict, List, Any

from data.coco_converter import CocoConverter
from data.file_system_manager import FileSystemManager
from data.augmentation_yolo import YoloAugmenter
from data.dataset_splitter import DatasetSplitter

from utils.config_logging import logger


class DatasetProcessor:

    """
    Encapsulates all dataset processing operations: 
    format conversion, splitting into train/val/test, and data augmentation.
    """


    def __init__ \
    (
        self, 
        paths: Dict[str, Path], 
        config: Dict[str, Path], 
        yolo_config: list[dict[str, str | Path]]
    ):
        """
        Initializes the processor with a dictionary of configuration paths.
        """
        self.paths = paths
        self.config = config
        self.yolo_config = yolo_config


    def _convert_coco_to_yolo(self) -> None:
        """
        Renames images, updates the COCO JSON, and converts annotations 
        from COCO format to YOLO format.
        """
        logger.info("Starting COCO to YOLO Conversion...")
        
        # 1. Rename Images and Update COCO JSON
        self.paths['renamed_images_dir'].mkdir(exist_ok=True)
        
        FileSystemManager.rename_and_copy_images(
            self.paths['unprocessed_images_dir'], 
            self.paths['renamed_images_dir'], 
            self.paths['original_names_map_file']
        )

        CocoConverter.update_filenames_in_coco_json(
            self.paths['original_names_map_file'], 
            self.paths['original_coco_json'], 
            self.paths['coco_json']
        )

        # 2. Prepare YOLO Label Directories
        self.paths['renamed_labels_dir'].mkdir(parents=True, exist_ok=True)
        FileSystemManager.clear_directory(self.paths['renamed_labels_dir'])

        # 3. Load and Convert Annotations
        try:
            with open(self.paths['coco_json'], 'r') as f:
                coco_data: Dict = json.load(f)

        except Exception as e:
            raise IOError(f"Error loading COCO JSON file: {e}")

        class_names: Dict[str, Any] = CocoConverter.get_filtered_class_names(coco_data)
        
        CocoConverter.convert_annotations(coco_data, self.paths['renamed_labels_dir'])
        CocoConverter.create_yaml_config(class_names, self.paths['yolo_base_dataset_yaml'])

        logger.info("COCO to YOLO conversion completed.")


    def _split_dataset(self) -> None:
        """
        Splits the annotated dataset into Train, Validation, and Test sets 
        using a stratified approach based on the dominant class.
        """
        logger.info("Starting Stratified Dataset Split...")        

        renamed_images: List[Path] = list(self.paths['renamed_images_dir'].glob('*.*'))
        logger.info(f"  > Total images found: {len(renamed_images)}")

        # Stratified Split
        splits: Dict[str, List[Path]] = DatasetSplitter.stratified_split(
            renamed_images, 
            self.paths['renamed_labels_dir']
        )

        # Save images and labels
        for split, images in splits.items():
            split_images_dir: Path = self.paths['images_dir'] / split
            split_labels_dir: Path = self.paths['labels_dir'] / split
            
            split_images_dir.mkdir(parents=True, exist_ok=True)
            split_labels_dir.mkdir(parents=True, exist_ok=True)

            # save_images_with_labels <-- saves the dataset while keeping the original renamed directories intact
            DatasetSplitter.organize_splits( #< moves images and labels into the specified split directories
                images, 
                self.paths['renamed_labels_dir'], 
                split_labels_dir, 
                split_images_dir, 
                split
            )
            logger.info(f"  > {split.capitalize()} set: {len(images)} images.")

        self.paths['renamed_labels_dir'].rmdir()
        self.paths['renamed_images_dir'].rmdir()

        logger.info("Dataset splitting completed.")


    def _augment_dataset(self) -> None:
        """
        Augments the training set and updates the 'train' path in the dataset YAML file.
        """
        logger.info("Starting Data Augmentation for the Train set...")
        
        self.paths['train_aug_images_dir'].mkdir(parents=True, exist_ok=True)
        self.paths['train_aug_labels_dir'].mkdir(parents=True, exist_ok=True)

        yolo_augmenter = YoloAugmenter(self.yolo_config)
        yolo_augmenter.augment_dataset(
            self.paths['train_images_dir'], self.paths['train_labels_dir'],
            self.paths['train_aug_images_dir'], self.paths['train_aug_labels_dir'],
            self.config['augmentations_per_image'],
            self.config['max_workers']
        )
        
        # Update or create a the YAML file to point to the augmented train set
        # CocoConverter.update_train_yaml(self.paths['yolo_aug_dataset_yaml'], 'images/train_augmented')
        CocoConverter.create_new_train_yaml(
            self.paths['yolo_base_dataset_yaml'], 
            self.paths['yolo_aug_dataset_yaml'], 
            'images/train_augmented'
        )
        logger.info("Data augmentation and YAML update completed.")


    def run_preprocessing_pipeline(self, augment: bool = False) -> None:
        """
        Executes the complete data preprocessing pipeline.
        """
        self._convert_coco_to_yolo()
        self._split_dataset()
        if augment:
            self._augment_dataset()
        logger.info("Data Preprocessing Pipeline Finished.")