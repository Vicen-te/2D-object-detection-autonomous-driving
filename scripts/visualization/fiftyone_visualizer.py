# fiftyone_visualizer.py
from argparse import ArgumentParser, Namespace
from enum import auto
from pathlib import Path
from typing import Dict, Optional
import fiftyone as fo
import fiftyone.types as fot
import json
import re
from tqdm import tqdm

from utils.config_logging import logger


class FiftyOneVisualizer:
    """
    A class to load, enrich, and visualize COCO or YOLO datasets using FiftyOne.
    """

    def __init__(
        self, 
        path: Path, 
        format: str, 
        split: Optional[str] = None, 
        names_map_file: Optional[Path] = None
    ):
        """
        Initialize visualizer with dataset parameters.

        Args:
            format: Dataset format ("yolo" or "coco").
            path: Path to the dataset root directory.
            split: Dataset split (required for YOLO format).
            names_map_path: Path to JSON file mapping original names (optional).
        """
        self.path: Path = path
        self.format: str = format
        self.split: Optional[str] = split
        self.names_map_file: Optional[Path] = names_map_file
        self.dataset: fo.Dataset = self.create_dataset(path, format, split)


    @staticmethod
    def load_coco_dataset(images_dir: Path, coco_json: Path) -> fo.Dataset:
        """
        Load a COCO dataset from a directory containing images and a JSON file with annotations.
        
        Args:
            images_path: Path to the directory containing images.
            labels_path: Path to the JSON file with COCO annotations.
            
        Returns:
            fo.Dataset: The loaded FiftyOne dataset.
        """
        logger.info(f"Loading COCO dataset from images: {images_dir}, labels: {coco_json}")
        
        dataset: fo.Dataset = fo.Dataset.from_dir(
            dataset_type=fot.COCODetectionDataset,
            data_path=str(images_dir),
            labels_path=str(coco_json),
            include_id=True,
        )
        dataset.compute_metadata()
        logger.info(f"COCO dataset loaded with {len(dataset)} samples.")
        return dataset


    @staticmethod
    def load_yolo_dataset(dataset_path: Path, yaml_path: Path, split: str) -> fo.Dataset:
        """
        Load a YOLO dataset from a directory and YAML file.
        
        Args:
            dataset_path: Path to the root directory containing the YOLO dataset structure (images/labels directories).
            yaml_path: Path to the YAML file defining the dataset structure and classes.
            split: The split of the dataset to load (e.g., 'train', 'val', 'test').
            
        Returns:
            fo.Dataset: The loaded FiftyOne dataset.
        """
        logger.info(f"Loading YOLO dataset (split: {split}) from root: {dataset_path}, config: {yaml_path}")
        
        dataset: fo.Dataset = fo.Dataset.from_dir(
            dataset_type=fot.YOLOv5Dataset, 
            dataset_dir=str(dataset_path), 
            yaml_path=str(yaml_path), 
            split=split
        )
        logger.info(f"YOLO dataset split '{split}' loaded with {len(dataset)} samples.")
        return dataset


    @staticmethod
    def add_original_names_field(dataset: fo.Dataset, json_path: Path) -> None:
        """
        Adds a persistent field to each sample in the dataset storing the original 
        (pre-renaming) image filename, handling augmented names (e.g., '_augX').
        
        Args:
            dataset: The FiftyOne dataset to modify.
            json_path: Path to the JSON file mapping original names to new names.
        """
        logger.info(f"\nEnriching dataset with original filenames from map: {json_path}")
        
        try:
            with open(json_path, 'r') as f:
                original_to_new: Dict[str, str] = json.load(f)

        except FileNotFoundError:
            logger.exception(f"Error: Name mapping JSON not found at {json_path}. Skipping original name enrichment.")
            return

        # Create inverse map in memory (new_name -> original_name)
        new_to_original: Dict[str, str] = {v: k for k, v in original_to_new.items()}

        # Create a view to iterate and update in batches (more efficient for large datasets)
        view = dataset.view()
        
        count_added = 0 

        for sample in tqdm(
            view, 
            desc='Adding original names', 
            unit='sample', 
            ncols=100
        ):
            filepath: Path = Path(sample.filepath)
            filename: str = filepath.name

            # 1. Clean the filename to find the base name (e.g., '00001_aug1.jpg' -> '00001.jpg')
            match: Optional[re.Match] = re.match(r"^(.*)_aug\d+\.(\w+)$", filename)
            base_name: str = f"{match.group(1)}.{match.group(2)}" if match else filename

            # 2. Look up the original name using the base name
            original_name: str = new_to_original.get(base_name)
            
            # Create a detached sample object for bulk update
            if original_name:
                sample["original_name"] = original_name
                sample.save()
                count_added += 1
            else:
                # Optional: still store UNKNOWN names
                sample["original_name"] = f"UNKNOWN:{filename}"
                sample.save()

        logger.info(f"Added 'original_name' field to {count_added} samples.")


    @classmethod
    def create_dataset(
        cls,
        path: Path,
        format: str,
        split: Optional[str] = None
    ) -> fo.Dataset:
        """
        Create and return a FiftyOne dataset based on the format.

        Args:
            format: Dataset format ("yolo" or "coco").
            path: Path to the dataset root directory.
            split: Dataset split (required for YOLO format).

        Returns:
            fo.Dataset: The loaded FiftyOne dataset.
        """
        dataset: fo.Dataset

        if format == "coco":
            images_dir: Path = path / "images"
            coco_json: Path = path / "labels_coco.json"
            if not images_dir.exists() or not coco_json.exists():
                raise FileNotFoundError(f"Required files/dirs for COCO not found in {path}")
            
            dataset = cls.load_coco_dataset(images_dir, coco_json)

        elif format == "yolo":
            if not split:
                raise ValueError("Split must be provided for YOLO format (e.g., 'train', 'val', 'test').")
            
            yolo_yaml: Path = path / "yolo_dataset.yaml"
            if not yolo_yaml.exists():
                # Common scenario: yaml is one level up
                yolo_yaml = path.parent / "yolo_dataset.yaml"
                if not yolo_yaml.exists():
                    raise FileNotFoundError(
                        f"YOLO YAML config not found at {path / 'yolo_dataset.yaml'} or {yolo_yaml}"
                    )
            
            dataset = cls.load_yolo_dataset(path, yolo_yaml, split)

        else:
            raise ValueError(f"Unsupported dataset format: {format}. Must be 'yolo' or 'coco'.")
        
        return dataset
    

    @classmethod
    def visualize_param(
        cls,
        path: Path, 
        format: str, 
        split: Optional[str] = None, 
        names_map_file: Optional[Path] = None
    ) -> None:
        """
        Loads and visualizes a dataset in the FiftyOne app.
        
        Args:
            format: Format of the dataset, either "yolo" or "coco".
            path: Path to the root directory of the dataset.
            split: Split of the dataset (only for YOLO format).
            names_map_file: Path to the original names JSON file (only for YOLO format, used for augmentation/renaming).
        """
        dataset: fo.Dataset = FiftyOneVisualizer.create_dataset(path, format, split)

        # Add original names field if the map path is provided
        if names_map_file: 
            FiftyOneVisualizer.add_original_names_field(dataset, names_map_file)
        
        logger.info("\nLaunching FiftyOne App... (This blocks until the app is closed)")
        session: fo.Session = fo.launch_app(dataset, auto = False)
        session.wait()
        logger.info("FiftyOne App closed.")


    def visualize(self) -> None:
        """
        Launch FiftyOne app using the stored parameters.
        """
        # Add original names field if the map path is provided
        if self.names_map_file: 
            self.add_original_names_field(self.dataset, self.names_map_file)
        
        logger.info("\nLaunching FiftyOne App... (This blocks until the app is closed)")
        session: fo.Session = fo.launch_app(self.dataset, auto = False)
        session.wait()
        logger.info("FiftyOne App closed.")
