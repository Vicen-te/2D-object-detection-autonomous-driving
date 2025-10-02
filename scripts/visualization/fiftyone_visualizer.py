# fiftyone_visualizer.py
from argparse import ArgumentParser, Namespace
from enum import auto
from pathlib import Path
from typing import Dict, List, Optional
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
        names_map_path: Optional[Path] = None
    ):
        """
        Initialize visualizer with dataset parameters.

        Args:
            format: Dataset format ("yolo" or "coco").
            path: Path to the dataset root folder.
            split: Dataset split (required for YOLO format).
            names_map_path: Path to JSON file mapping original names (optional).
        """
        self.path: Path = path
        self.format: str = format
        self.split: Optional[str] = split
        self.names_map_path: Optional[Path] = names_map_path
        self.dataset: fo.Dataset = self.create_dataset(path, format, split)


    @staticmethod
    def load_coco_dataset(images_path: Path, labels_path: Path) -> fo.Dataset:
        """
        Load a COCO dataset from a directory containing images and a JSON file with annotations.
        
        Args:
            images_path: Path to the directory containing images.
            labels_path: Path to the JSON file with COCO annotations.
            
        Returns:
            fo.Dataset: The loaded FiftyOne dataset.
        """
        logger.info(f"Loading COCO dataset from images: {images_path}, labels: {labels_path}")
        
        dataset: fo.Dataset = fo.Dataset.from_dir(
            dataset_type=fot.COCODetectionDataset,
            data_path=str(images_path),
            labels_path=str(labels_path),
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
            dataset_path: Path to the root directory containing the YOLO dataset structure (images/labels folders).
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
            logger.info(f"Error: Name mapping JSON not found at {json_path}. Skipping original name enrichment.")
            return

        # Create inverse map in memory (new_name -> original_name)
        new_to_original: Dict[str, str] = {v: k for k, v in original_to_new.items()}

        # Create a view to iterate and update in batches (more efficient for large datasets)
        view = dataset.view()
        
        updates: List[fo.Sample] = []

        for sample in tqdm(
            view, 
            desc='Adding original names', 
            unit='sample', 
            ncols=100
        ):
            filepath: Path = Path(sample.filepath)
            filename: str = filepath.name

            # 1. Clean the filename to find the base name (e.g., '00001_aug1.jpg' -> '00001.jpg')
            base_name: str
            match: Optional[re.Match] = re.match(r"^(.*)_aug\d+\.(\w+)$", filename)
            
            if match:
                # Reconstruct base name (e.g., 00001.jpg)
                base_name = f"{match.group(1)}.{match.group(2)}"
            else:
                base_name = filename

            # 2. Look up the original name using the base name
            original_name: str = new_to_original.get(base_name, f"UNKNOWN: {filename}")
            
            # Create a detached sample object for bulk update
            sample["original_name"] = original_name
            updates.append(sample)

        # Apply updates to the dataset
        dataset.save_samples(updates)
        logger.info(f"Added 'original_name' field to {len(updates)} samples.")


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
            path: Path to the dataset root folder.
            split: Dataset split (required for YOLO format).

        Returns:
            fo.Dataset: The loaded FiftyOne dataset.
        """
        dataset: fo.Dataset

        if format == "coco":
            images_path: Path = path / "images"
            coco_path: Path = path / "labels_coco.json"
            if not images_path.exists() or not coco_path.exists():
                raise FileNotFoundError(f"Required files/dirs for COCO not found in {path}")
            
            dataset = cls.load_coco_dataset(images_path, coco_path)

        elif format == "yolo":
            if not split:
                raise ValueError("Split must be provided for YOLO format (e.g., 'train', 'val', 'test').")
            
            yolo_yaml_path: Path = path / "yolo_dataset.yaml"
            if not yolo_yaml_path.exists():
                # Common scenario: yaml is one level up
                yolo_yaml_path = path.parent / "yolo_dataset.yaml"
                if not yolo_yaml_path.exists():
                    raise FileNotFoundError(
                        f"YOLO YAML config not found at {path / 'yolo_dataset.yaml'} or {yolo_yaml_path}"
                    )
            
            dataset = cls.load_yolo_dataset(path, yolo_yaml_path, split)

        else:
            raise ValueError(f"Unsupported dataset format: {format}. Must be 'yolo' or 'coco'.")
        
        return dataset
    

    @classmethod
    def visualize_param(
        cls,
        path: Path, 
        format: str, 
        split: Optional[str] = None, 
        names_map_path: Optional[Path] = None
    ) -> None:
        """
        Loads and visualizes a dataset in the FiftyOne app.
        
        Args:
            format: Format of the dataset, either "yolo" or "coco".
            path: Path to the root folder of the dataset.
            split: Split of the dataset (only for YOLO format).
            names_map_path: Path to the original names JSON file (only for YOLO format, used for augmentation/renaming).
        """
        dataset: fo.Dataset = FiftyOneVisualizer.create_dataset(path, format, split)

        # Add original names field if the map path is provided
        if names_map_path: 
            FiftyOneVisualizer.add_original_names_field(dataset, names_map_path)
        
        logger.info("\nLaunching FiftyOne App... (This blocks until the app is closed)")
        session: fo.Session = fo.launch_app(dataset, auto = False)
        session.wait()
        logger.info("FiftyOne App closed.")


    def visualize(self) -> None:
        """
        Launch FiftyOne app using the stored parameters.
        """
        # Add original names field if the map path is provided
        if self.names_map_path: 
            self.add_original_names_field(self.dataset, self.names_map_path)
        
        logger.info("\nLaunching FiftyOne App... (This blocks until the app is closed)")
        session: fo.Session = fo.launch_app(self.dataset, auto = False)
        session.wait()
        logger.info("FiftyOne App closed.")


    @staticmethod
    def parse_arguments() -> Namespace:
        """
        Parse command line arguments for the script.
        
        Returns:
            Namespace: Parsed command line arguments.
        """ 
        parser: ArgumentParser = ArgumentParser(description="Visualize NN/CV project dataset with FiftyOne")
        parser.add_argument("--path", "--p", type=Path, required=True, help="Path to the root folder of the dataset.")
        parser.add_argument("--format", "--f", type=str, choices=["yolo", "coco"], required=True, help="Format of the dataset ('yolo' or 'coco').")
        parser.add_argument("--split", "--s", required=False, type=str, choices=["train", "val", "test"], help="Split of the dataset (required for YOLO format).")
        parser.add_argument("--names", "--n", required=False, type=Path, help="Path to the original names JSON file (e.g., original_names_map.json).")
        args: Namespace = parser.parse_args()
        return args


if __name__ == "__main__":
    try:
        args: Namespace = FiftyOneVisualizer.parse_arguments()
        FiftyOneVisualizer.visualize_param(
            path=args.path, 
            format=args.format, 
            split=args.split, 
            names_map_path=args.names
        )

    except Exception as e:
        logger.info(f"An error occurred during visualization: {e}")