# dataset_splitter.py
import math
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
import shutil
from collections import Counter
from tqdm import tqdm

from utils.config_logging import setup_logging
logger = setup_logging()


class DatasetSplitter:

    """
    A class for splitting a YOLO-format dataset (images and labels) into 
    train, validation, and test sets using stratification based on the 
    dominant class in each image.
    """


    @staticmethod
    def map_images_to_dominant_class(image_files: List[Path], labels_path: Path) -> Dict[Path, int]:
        """
        Maps each image to its dominant class label based on YOLO label files.
        
        Args:
            image_files: List of image file paths.
            labels_path: Directory containing YOLO label files.

        Returns:
            Dict[Path, int]: Dictionary mapping each image file path to its dominant class ID.
        """
        image_to_class: Dict[Path, int] = {}
        
        logger.info("Mapping images to dominant class for stratification...")

        for image_file in tqdm(
            image_files, 
            desc='Mapping classes', 
            unit='image', 
            ncols=100
        ):
            label_path: Path = labels_path / (image_file.stem + '.txt')
            
            if not label_path.exists():
                continue # Skip images without annotation

            try:
                with label_path.open('r') as f:
                    lines: List[str] = f.readlines()
                
                # Extract all class IDs present in the file
                classes: List[int] = []
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        classes.append(int(parts[0]))
                
                if classes:
                    # Determine the dominant class (most common class ID)
                    dominant_class: int = Counter(classes).most_common(1)[0][0] 
                    image_to_class[image_file] = dominant_class
            except Exception as e:
                logger.info(f"Warning: Could not process label file {label_path}. Skipping. Error: {e}")

        logger.info(f"Successfully mapped {len(image_to_class)} images to a dominant class.")
        return image_to_class


    @classmethod
    def stratified_split(
        cls,
        images_path: Path,
        labels_path: Path,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Splits image paths into train, validation, and test sets using stratified sampling 
        based on the dominant class of each image.

        Approach:
            1. Split into Train vs Test (train_ratio, test_ratio)
            2. Split Train into Train vs Validation (train_ratio adjusted, val_ratio)
        
        Args:
            images_path: Directory containing the images.
            labels_path: Directory containing the YOLO label files.
            train_ratio: Proportion of data for the training set (e.g., 0.70).
            val_ratio: Proportion of data for the validation set (e.g., 0.15).
            test_ratio: Proportion of data for the test set (e.g., 0.15).
            random_state: Seed for reproducibility.

        Returns:
            Tuple[List[Path], List[Path], List[Path]]: Train, validation, and test splits.
        """
        # Ensure ratios sum to 1.0 (or close)
        if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Train, validation, and test ratios must sum to 1.0.")

        # Map images to their dominant class
        image_to_class: Dict[Path, int] = cls.map_images_to_dominant_class(images_path, labels_path)
        
        # Filter images that actually have labels (and thus a dominant class)
        stratify_images: List[Path] = list(image_to_class.keys())
        stratify_classes: List[int] = [image_to_class[img] for img in stratify_images]

        if not stratify_images:
            logger.info("No labeled images found. Returning empty splits.")
            return ([], [], [])
        
        # --- Split 1: Train vs Test ---
        train_images, temp_images, _, temp_classes = train_test_split(
            stratify_images, 
            stratify_classes, 
            test_size=test_ratio, 
            random_state=random_state,
            stratify=stratify_classes
        )

        # --- Split 2: Train vs Validation ---
        # Adjust validation ratio relative to the remaining train set
        relative_test_size: float = val_ratio / train_ratio
        
        val_images, test_images, _, _ = train_test_split(
            temp_images, 
            temp_classes, 
            test_size=relative_test_size, 
            random_state=random_state,
            stratify=temp_classes
        )
        
        logger.info(f"\nDataset split complete:")
        logger.info(f"  Train: {len(train_images)} images ({len(train_images)/len(stratify_images):.2%})")
        logger.info(f"  Validation: {len(val_images)} images ({len(val_images)/len(stratify_images):.2%})")
        logger.info(f"  Test: {len(test_images)} images ({len(test_images)/len(stratify_images):.2%})")

        return train_images, val_images, test_images


    @staticmethod
    def save_images_with_labels(
        images: List[Path],
        input_labels_path: Path,
        output_labels_path: Path,
        output_images_path: Path,
        split_name: str
    ) -> None:
        """
        Creates output directories and copies images and corresponding label files, 
        showing progress using tqdm.

        Args:
            images: List of image file paths to process.
            input_labels_path: Directory containing the original label files.
            output_labels_path: Base directory to save YOLO label files (e.g., 'labels/train').
            output_images_path: Base directory to save image files (e.g., 'images/train').
            split_name: Name of the data split ('train', 'val', or 'test') for progress display.
        """
        
        # Create output directories for the split
        output_images_path.mkdir(parents=True, exist_ok=True)
        output_labels_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nSaving {len(images)} files to {split_name} split...")

        for image_file in tqdm(
            images, 
            desc=f'Saving {split_name} data', 
            unit='file', 
            ncols=100
        ):
            label_file: str = image_file.stem + '.txt'
            input_label_path = input_labels_path / label_file
            
            # Since we only include images with labels in the stratified_split output, 
            # this check mainly handles edge cases where an image was included but label was deleted.
            if not input_label_path.exists():
                continue 

            # Prepare paths
            input_image_path: Path = image_file # image_file is already the full path
            destination_label_path: Path = output_labels_path / label_file
            destination_image_path: Path = output_images_path / image_file.name

            # Copy annotation file
            shutil.copy2(input_label_path, destination_label_path)

            # Copy image file
            shutil.copy2(input_image_path, destination_image_path)
            
        logger.info(f"{split_name.capitalize()} split saved.")
