from pathlib import Path
from typing import List, Dict, Tuple
from progress_bar import printProgressBar
from sklearn.model_selection import train_test_split
import shutil
from collections import Counter



def map_images_to_dominant_class(image_files: List[Path], labels_path: Path) -> Dict[Path, int]:
    """
    Maps each image to its dominant class label based on YOLO label files.

    Args:
        image_files (List[Path]): List of image file paths.
        labels_path (Path): Directory containing YOLO label files.

    Returns:
        Dict[Path, int]: Dictionary mapping each image file to its dominant class ID.
    """
    total: int = len(image_files)

    # Initial call to print 0% progress
    printProgressBar(0, total, prefix = 'Checking Progress:', suffix = 'Complete', length = 50)

    # Map image to its dominant class
    image_to_class: Dict[Path, int] = {}
    for i, image_file in enumerate(image_files, start=1):
        label_path: Path = labels_path / (image_file.stem + '.txt')
        
        if not label_path.exists():
            continue  #< Skip images without annotation

        with label_path.open('r') as f:
            lines: List[str] = f.readlines()
            classes: List[int] = [int(line.split()[0]) for line in lines if line.strip()]
            
            if classes:
                # Dominant-class stratification
                dominant_class: int = Counter(classes).most_common(1)[0][0] 
                image_to_class[image_file] = dominant_class

        # Update Progress Bar
        if i * 100 // total != (i-1) * 100 // total:
            printProgressBar(i, total, prefix='Checking Progress:', suffix='Complete', length=50)

    return image_to_class



def stratified_split(images: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Splits image paths into train, validation, and test sets with fixed proportions.

    Args:
        images (List[Path]): List of image file paths.

    Returns:
        Tuple[List[Path], List[Path], List[Path]]: Train, validation, and test splits.
    """

    train_imgs: List[Path]
    temp_imgs: List[Path]
    val_imgs: List[Path]
    test_imgs: List[Path]

    # First, split into train and temp (val + test)
    train_imgs, temp_imgs = train_test_split(images, test_size=0.2, random_state=42)

    # Then split temp into val and test (50% each of the 20% = 10% and 10%)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    # Stratified split: returns (train_images, val_images, test_images)
    return train_imgs, val_imgs, test_imgs



def save_images_with_labels(
    images: List[Path],
    input_labels_path: Path,
    input_images_path: Path,
    output_labels_path: Path,
    output_images_path: Path,
    split_name: str
) -> None:
    """
    Copies images and corresponding label files to target directories, showing progress.

    Args:
        images (List[Path]): List of image file paths to process.
        input_labels_path (Path): Directory containing the original label files.
        input_images_path (Path): Directory containing the original images.
        output_labels_path (Path): Directory to save YOLO label files.
        output_images_path (Path): Directory to save image files.
        split_name (str): Name of the data split ('train', 'val', or 'test') for progress display.
    """
    total: int = len(images)

    # Initial call to print 0% progress
    printProgressBar(0, total, prefix = f'{split_name.capitalize()} Saving Progress:', suffix = 'Complete', length = 50)

    for i, image_file in enumerate(images, start=1):
        label_file: str = image_file.stem + '.txt'
        source_label_path = input_labels_path / label_file

        if not source_label_path.exists():
            continue  #< Skip if no annotation

        source_image_path: Path = input_images_path / image_file.name
        destination_label_path: Path  = output_labels_path / label_file
        destination_image_path: Path  = output_images_path / image_file.name

        # Copy annotation content
        label_content: str  = source_label_path.read_text()
        destination_label_path.write_text(label_content)

         # Copy image using shutil (faster and lower overhead)
        shutil.copy2(source_image_path, destination_image_path)

        # Update progress bar on each percent change
        if i * 100 // total != (i-1) * 100 // total:
            printProgressBar(i, total, prefix = f'{split_name.capitalize()} Saving Progress:', suffix = 'Complete', length=50)