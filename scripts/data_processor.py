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


    def __init__(self, paths: Dict[str, Path]):
        """
        Initializes the processor with a dictionary of configuration paths.
        """
        self.paths = paths


    def _convert_coco_to_yolo(self) -> None:
        """
        Renames images, updates the COCO JSON, and converts annotations 
        from COCO format to YOLO format.
        """
        logger.info("Starting COCO to YOLO Conversion...")
        
        # 1. Rename Images and Update COCO JSON
        self.paths['renamed_images_path'].mkdir(exist_ok=True)
        FileSystemManager.rename_and_copy_images(
            self.paths['unprocessed_images_path'], 
            self.paths['renamed_images_path'], 
            self.paths['original_names_map']
        )
        CocoConverter.update_coco_json_filenames(
            self.paths['original_names_map'], 
            self.paths['original_coco_json_file'], 
            self.paths['coco_json_file']
        )

        # 2. Prepare YOLO Label Directories
        FileSystemManager.clear_directory(self.paths['renamed_labels_path'])
        self.paths['renamed_labels_path'].mkdir(parents=True, exist_ok=True)

        # 3. Load and Convert Annotations
        try:
            with open(self.paths['coco_json_file'], 'r') as f:
                coco_data: Dict = json.load(f)
        except Exception as e:
            raise IOError(f"Error loading COCO JSON file: {e}")

        class_names: Dict[str, Any] = CocoConverter.get_filtered_class_names(coco_data)
        
        CocoConverter.convert_annotations(coco_data, self.paths['renamed_labels_path'])
        CocoConverter.create_yaml_config(class_names, self.paths['yolo_dataset_path'])
        logger.info("COCO to YOLO conversion completed.")


    def _split_dataset(self) -> None:
        """
        Splits the annotated dataset into Train, Validation, and Test sets 
        using a stratified approach based on the dominant class.
        """
        logger.info("Starting Stratified Dataset Split...")

        renamed_images_path = self.paths['renamed_images_path']
        renamed_labels_path = self.paths['renamed_labels_path']
        
        all_image_paths: List[Path] = list(renamed_images_path.glob('*.*'))
        logger.info(f"  > Total images found: {len(all_image_paths)}")

        # Map images to their dominant class
        image_dominant_class_map: Dict[Path, int] = DatasetSplitter.map_images_to_dominant_class(
            all_image_paths, renamed_labels_path
        )

        if not image_dominant_class_map:
            raise Exception("No annotated images found for splitting.")

        # Stratified Split
        train_images, val_images, test_images = DatasetSplitter.stratified_split(all_image_paths, renamed_labels_path)

        split_data: Dict[str, List[str]] = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }

        # Save images and labels
        for split, images in split_data.items():
            split_images_path: Path = self.paths['images_path'] / split
            split_labels_path: Path = self.paths['labels_path'] / split
            
            split_images_path.mkdir(parents=True, exist_ok=True)
            split_labels_path.mkdir(parents=True, exist_ok=True)

            DatasetSplitter.save_images_with_labels(
                images, 
                renamed_labels_path, 
                split_labels_path, 
                split_images_path, 
                split
            )
            logger.info(f"  > {split.capitalize()} set: {len(images)} images.")

        logger.info("Dataset splitting completed.")


    def _augment_dataset(self, augmentations_per_image: int = 3, max_workers: int = 8) -> None:
        """
        Augments the training set and updates the 'train' path in the dataset YAML file.
        """
        logger.info("Starting Data Augmentation for the Train set...")
        
        train_images_path = self.paths['images_path'] / 'train'
        train_labels_path = self.paths['labels_path'] / 'train'
        train_aug_images_path = self.paths['train_aug_images_path']
        train_aug_labels_path = self.paths['train_aug_labels_path']

        train_aug_images_path.mkdir(parents=True, exist_ok=True)
        train_aug_labels_path.mkdir(parents=True, exist_ok=True)

        yolo_augmenter = YoloAugmenter()
        yolo_augmenter.augment_dataset(
            train_images_path, train_labels_path,
            train_aug_images_path, train_aug_labels_path,
            augmentations_per_image=augmentations_per_image,
            max_workers=max_workers
        )
        
        # Update or create a the YAML file to point to the augmented train set
        # CocoConverter.update_train_path(self.paths['yolo_dataset_path'], 'images/train_augmented')
        CocoConverter.create_new_train_yaml(
            self.paths['yolo_dataset_path'], 
            self.paths['yolo_dataset_path2'], 
            'images/train_augmented'
        )
        logger.info("Data augmentation and YAML update completed.")


    def run_preprocessing_pipeline(self, augment: bool = True) -> None:
        """
        Executes the complete data preprocessing pipeline.
        """
        self._convert_coco_to_yolo()
        self._split_dataset()
        if augment:
            self._augment_dataset(
                augmentations_per_image=self.paths.get('augmentations_per_image', 3),
                max_workers=self.paths.get('max_workers', 8)
            )
        logger.info("Data Preprocessing Pipeline Finished.")