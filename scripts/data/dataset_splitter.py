# dataset_splitter.py
import math
from pathlib import Path
import random
from typing import List, Dict, Tuple
from sklearn.model_selection import StratifiedShuffleSplit
import shutil
from collections import Counter
from tqdm import tqdm
from utils.config_logging import logger


class DatasetSplitter:

    """
    A class for splitting a YOLO-format dataset (images and labels) into 
    train, validation, and test sets using stratification based on the 
    dominant class in each image.
    """


    @staticmethod
    def map_images_to_dominant_class(image_files: List[Path], labels_dir: Path) -> Dict[Path, int]:
        """
        Maps each image to its dominant class label based on YOLO label files.

        Args:
            image_files: List of image file paths.
            labels_dir: Directory containing YOLO label files.

        Returns:
            Dict[Path, int]: Dictionary mapping each image path to its dominant class ID.
        """
        image_to_class: Dict[Path, int] = {}
        logger.info("Mapping images to dominant class for stratification...")

        for image_file in tqdm(
            image_files, 
            desc='Mapping classes', 
            unit='image', 
            ncols=100
        ):
            label_file: Path = labels_dir / (image_file.stem + '.txt')

            try:
                # Extract all class IDs present in the file
                with label_file.open('r') as f:
                    classes = [int(line.strip().split()[0]) for line in f if line.strip()]
                
                if classes:
                    # Determine the dominant class (most common class ID)
                    dominant_class: int = Counter(classes).most_common(1)[0][0] 
                    image_to_class[image_file] = dominant_class
                else:
                    image_to_class[image_file] = -1

            except Exception as e:
                logger.exception(f"Could not process label file {label_file}. Skipping. Exception: {e}")

        logger.info(f"Successfully mapped {len(image_to_class)} images to a dominant class.")
        return image_to_class


    @classmethod
    def stratified_split(
        cls,
        images_dir: Path,
        labels_dir: Path,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ) -> Dict[str, List[Path]]:
        """
        Splits image paths into train, validation, and test sets using stratified sampling 
        based on the dominant class of each image.

        Args:
            images_dir: Directory containing the images.
            labels_dir: Directory containing the YOLO label files.
            train_ratio: Proportion of data for the training set (e.g., 0.70).
            val_ratio: Proportion of data for the validation set (e.g., 0.15).
            test_ratio: Proportion of data for the test set (e.g., 0.15).
            random_state: Seed for reproducibility.

        Returns:
            Dict[str, List[Path]]: Dictionary with keys 'train', 'val', 'test', 
            each mapping to a list of image file paths.
        """
        # Ensure ratios sum to 1.0 (or close)
        if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Train, validation, and test ratios must sum to 1.0.")

        # Map images to their dominant class
        image_to_class: Dict[Path, int] = cls.map_images_to_dominant_class(images_dir, labels_dir)

        if not image_to_class:
            raise Exception("No annotated images found for splitting.")
        
        # Separate images with classes and without classes
        images_with_classes = [img for img, cls in image_to_class.items() if cls != -1]
        classes_for_stratify = [cls for cls in image_to_class.values() if cls != -1]
        images_without_classes = [img for img, cls in image_to_class.items() if cls == -1]

        if not images_with_classes:
            logger.error("No labeled images found. Returning empty splits.")
            return {"train": [], "val": [], "test": []}
        
        # --- Split 1: Test vs Temp(Train + Validation) ---
        sss1 = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=(val_ratio + test_ratio), 
            random_state=random_state
        )
        train_idx, temp_idx = next(sss1.split(images_with_classes, classes_for_stratify))

        train_images = [images_with_classes[i] for i in train_idx]
        temp_images = [images_with_classes[i] for i in temp_idx]
        temp_classes = [classes_for_stratify[i] for i in temp_idx]

        # --- Split 2: Train vs Validation ---
        # Adjust validation ratio relative to the remaining train set
        val_relative_ratio : float = val_ratio / (val_ratio + test_ratio)

        sss2 = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=val_relative_ratio, 
            random_state=random_state
        )
        val_idx, test_idx = next(sss2.split(temp_images, temp_classes))

        val_images = [temp_images[i] for i in val_idx]
        test_images = [temp_images[i] for i in test_idx]

        # Randomly distribute images without classes according to the ratios
        if images_without_classes:
            random.seed(random_state)
            random.shuffle(images_without_classes)

            n = len(images_without_classes)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_images += images_without_classes[:n_train]
            val_images += images_without_classes[n_train:n_train+n_val]
            test_images += images_without_classes[n_train+n_val:]
        
        logger.info(f"Dataset split complete:")
        logger.info(f"  Train: {len(train_images)} images ({len(train_images)/len(image_to_class):.2%})")
        logger.info(f"  Validation: {len(val_images)} images ({len(val_images)/len(image_to_class):.2%})")
        logger.info(f"  Test: {len(test_images)} images ({len(test_images)/len(image_to_class):.2%})")
        
        splits: Dict[str, List[Path]] = {
            "train": train_images,
            "val": val_images,
            "test": test_images,
        }

        return splits


    @staticmethod
    def save_images_with_labels(
        images: List[Path],
        input_labels_dir: Path,
        output_labels_dir: Path,
        output_images_dir: Path,
        split_name: str
    ) -> None:
        """
        Creates output directories and copies images and corresponding label files,
        showing progress using tqdm.

        Args:
            images: List of image file paths to process.
            input_labels_dir: Directory containing the original label files.
            output_labels_dir: Directory to save YOLO label files (e.g., 'labels/train').
            output_images_dir: Directory to save image files (e.g., 'images/train').
            split_name: Name of the data split ('train', 'val', or 'test') for progress display.
        """
        logger.info(f"Saving {len(images)} files to {split_name} split...")

        for image_file in tqdm(
            images, 
            desc=f'Saving {split_name} data', 
            unit='file', 
            ncols=100
        ):
            label_file: str = f"{image_file.stem}.txt"
            input_label = input_labels_dir / label_file
            
            # Since we only include images with labels in the stratified_split output, 
            # this check mainly handles edge cases where an image was included but label was deleted.
            if not input_label.exists():
                continue 

            # Prepare paths
            destination_label_file: Path = output_labels_dir / label_file
            destination_image_file: Path = output_images_dir / image_file.name

            # Copy annotation file
            shutil.copy2(input_label, destination_label_file)

            # Copy image file
            shutil.copy2(image_file, destination_image_file)
            
        logger.info(f"{split_name.capitalize()} split saved.")

        
    @staticmethod
    def organize_splits(
        images: List[Path],
        input_labels_dir: Path,
        output_labels_dir: Path,
        output_images_dir: Path,
        split_name: str
    ) -> None:
        """
        Organizes images and labels into train/val/test directories by moving files
        from the original directories to the specified output directories.

        Args:
            images: List of image file paths to organize.
            input_labels_dir: Directory containing the original label files.
            output_labels_dir: Target directory to save the label files.
            output_images_dir: Target directory to save the image files.
            split_name: Name of the dataset split ('train', 'val', or 'test').
        """
      
        for image_file in tqdm(
            images, 
            desc=f'Saving {split_name} data', 
            unit='file', 
            ncols=100
        ):
            src_image = image_file.parent / image_file.name
            src_label = input_labels_dir / f"{image_file.stem}.txt"
            
            dst_image = output_images_dir / image_file.name
            dst_label = output_labels_dir / f"{image_file.stem}.txt"

            if src_image.exists():
                src_image.rename(dst_image)
            else:
                logger.warning(f"Image not found: {src_image}")

            if src_label.exists():
                src_label.rename(dst_label)
            else:
                logger.warning(f"Label not found: {src_label}")


    @classmethod
    def revert_organization(
        cls,
        images_dir: Path,
        labels_dir: Path,
        splits: List[str] = ["train", "val", "test"],
        image_ext: str = ".jpg",
        label_ext: str = ".txt"
    ):
        """
        Reverts the dataset organization from split directories back to the base directories:
        - Moves images from split subdirectories back to the main images directory.
        - Moves labels from split subdirectories back to the main labels directory.
        - Attempts to remove empty split directories.

        Args:
            images_dir: Base directory containing images.
            labels_dir: Base directory containing labels.
            splits: List of split directory names (default: ["train", "val", "test"]).
            image_ext: Image file extension (default: ".jpg").
            label_ext: Label file extension (default: ".txt").
        """
        logger.info("Reverting images organization...")
        cls._move_split_files_to_base(splits, images_dir, image_ext)

        logger.info("Reverting labels organization...")
        cls._move_split_files_to_base(splits, labels_dir, label_ext)

        logger.info("Organization reverted: images and labels moved back to base directory.")


    @staticmethod
    def _move_split_files_to_base(
        dir: Path,
        split_names: List[str],
        ext: str
    ):
        """
        Moves files from split subdirectories back to the base directory.

        Args:
            dir: Base directory containing the split subdirectories.
            split_names: List of split directory names (e.g., ["train", "val", "test"]).
            ext: File extension to move (e.g., ".jpg" or ".txt").
        """
        for split_name in split_names:
            split_dir = dir / split_name

            if not split_dir.exists():
                continue

            for file_path in tqdm(
                list(split_dir.glob(f"*{ext}")), 
                desc=f"{split_name} {ext}",
                unit='file', 
                ncols=100
            ):
                dst_path = dir / file_path.name
                file_path.rename(dst_path)

            try:
                split_dir.rmdir()
            except OSError:
                logger.exception(f"Could not remove directory {split_dir}, it is not empty.")