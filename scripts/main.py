import json, time 
from pathlib import Path
from typing import Any, Dict, List

from convert_coco_yolo import (
    convert_labels, create_yaml_from_coco, 
    get_filtered_categories_and_names, update_train_path, update_coco_json
)

from utils import clear_directory, rename_images
from split_dataset import map_images_to_dominant_class, stratified_split, save_images_with_labels
from visualize_dataset import visualize_dataset
from augmentation_yolo import augment_dataset
from train_model import train_model, evaluate_model


def convert_coco_to_yolo(
    unprocessed_images_path: Path,
    renamed_images_folder: Path,
    original_coco_json_file: Path,
    coco_json_file: Path,
    original_names_map: Path,
    renamed_labels_path: Path,
    yaml_output_path: Path
) -> None:
    """
    Convert COCO annotations to YOLO format and prepare the dataset for training.
    Args:
        unprocessed_images_path (Path): Path to the folder containing unprocessed images.
        renamed_images_folder (Path): Path to the folder where renamed images will be stored.
        original_coco_json_file (Path): Path to the original COCO JSON file.
        coco_json_file (Path): Path to the output COCO JSON file in YOLO format.
        original_names_map (Path): Path to the JSON file mapping original names to new names.
        renamed_labels_path (Path): Directory where YOLO annotations will be saved.
        yaml_output_path (Path): Path to save the YAML configuration file for YOLO dataset.
    """
    
    print("Renaming images and updating COCO JSON...")
    renamed_images_folder.mkdir(exist_ok=True)
    rename_images(unprocessed_images_path, renamed_images_folder, original_names_map)
    update_coco_json(original_names_map, original_coco_json_file, coco_json_file)

    # Clear the directory for renamed labels
    clear_directory(renamed_labels_path)
    # Create the directory for renamed labels if it doesn't exist
    renamed_labels_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load COCO annotations
        with open(coco_json_file, 'r') as f:
            coco_data: Dict = json.load(f)

    except Exception as e:
        print(f"Error loading COCO JSON file: {e}")
        return

    filtered_categories: Dict[int, str] 
    image_names: List[Dict[str, Any]]
    filtered_categories , image_names = get_filtered_categories_and_names(coco_data)
    
    convert_labels(coco_data, filtered_categories, renamed_labels_path)
    create_yaml_from_coco(image_names, yaml_output_path)



def split_dataset(
    images_path: Path,
    labels_path: Path,
    renamed_images_folder: Path,
    renamed_labels_path: Path
) -> None:
    """
    Split the dataset into train, validation, and test sets, and save images and labels accordingly.
    Args:
        images_path (Path): Path to the folder containing images.
        labels_path (Path): Path to the folder containing labels.
        renamed_images_folder (Path): Path to the folder where renamed images are stored.
        renamed_labels_path (Path): Path to the folder where renamed labels are stored.
    """

    # Get list of all image file paths
    all_image_paths: List[Path] = [f for ext in ('*.jpg', '*.jpeg', '*.png') for f in renamed_images_folder.glob(ext)]
    print(f"Found {len(all_image_paths)} images")

    # Map images to their dominant class based on annotations
    image_dominant_class_map: Dict = map_images_to_dominant_class(all_image_paths, renamed_labels_path)

    if not image_dominant_class_map:
       raise Exception("No images with annotations found in source directory.")

    # List of images that have annotations (for splitting)
    annotated_image_paths: List[str] = list(image_dominant_class_map.keys())

    train_images: List[str]
    val_images: List[str]
    test_images: List[str]

    # Split the dataset into train, validation, and test sets
    print("Stratifying dataset...")
    train_images, val_images, test_images = stratified_split(annotated_image_paths)

    # Associate each split with its respective list of images
    split_data: Dict[str, List[str]]  = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }

    for split, images in split_data.items():
        split_images_path: Path = images_path / split
        split_labels_path: Path = labels_path / split

        # Create 'images' and 'labels' directories if they don't exist
        split_images_path.mkdir(parents=True, exist_ok=True)
        split_labels_path.mkdir(parents=True, exist_ok=True)

        # Save images and corresponding labels to their respective folders
        save_images_with_labels(
            images, 
            renamed_labels_path, 
            renamed_images_folder, 
            split_labels_path, 
            split_images_path, 
            split
        )

    print(f"Total images: {len(images)}")

    for split, images in split_data.items():
        print(f"{split.capitalize()}: {len(images)}")



def create_data(
    images_path: Path,
    labels_path: Path,
    renamed_images_path: Path,
    renamed_labels_path: Path,
    original_coco_json_file: Path,
    coco_json_file: Path,
    original_names_map: Path,
    unprocessed_images_path: Path,
    yolo_dataset_path: Path,
    train_images_path: Path,
    train_labels_path: Path,
    train_aug_images_path: Path,
    train_aug_labels_path: Path
) -> None:
    """
    Create the dataset by renaming images, converting COCO annotations to YOLO format,
    splitting the dataset into train, validation, and test sets, and augmenting the training data.
    Args:
        images_path (Path): Path to the folder containing images.
        labels_path (Path): Path to the folder containing labels.
        renamed_images_path (Path): Path to the folder where renamed images will be stored.
        renamed_labels_path (Path): Path to the folder where renamed labels will be stored.
        original_coco_json_file (Path): Path to the original COCO JSON file.
        coco_json_file (Path): Path to the output COCO JSON file in YOLO format.
        original_names_map (Path): Path to the JSON file mapping original names to new names.
        unprocessed_images_path (Path): Path to the folder containing unprocessed images.
        yolo_dataset (Path): Path to save the YAML configuration file for YOLO dataset.
        train_images_path (Path): Path to the folder for training images.
        train_labels_path (Path): Path to the folder for training labels.
        train_aug_images_path (Path): Path to the folder for augmented training images.
        train_aug_labels_path (Path): Path to the folder for augmented training labels.
    """
    
    convert_coco_to_yolo(unprocessed_images_path, renamed_images_path, original_coco_json_file,
                         coco_json_file, original_names_map, renamed_labels_path, yolo_dataset_path)
    split_dataset(images_path, labels_path, renamed_images_path, renamed_labels_path)
    
    train_aug_images_path.mkdir(parents=True, exist_ok=True)
    train_aug_labels_path.mkdir(parents=True, exist_ok=True)
    augment_dataset(
        train_images_path, train_labels_path,
        train_aug_images_path, train_aug_labels_path,
        augmentations_per_image=3,
        max_workers=8
    )
    update_train_path(yolo_dataset_path, 'images/train_augmented')



def train_models(
    yamls_path: Path, 
    data_yml_path: Path,
    train_results_path: Path, 
    val_results_path: Path
) -> None:
    """
    Train YOLO models using different configurations and evaluate them.
    Args:
        yamls_path (Path): Path to the directory containing YAML configuration files.
        data_yml_path (Path): Path to the YOLO dataset YAML file.
        train_results_path (Path): Path to save training results.
        val_results_path (Path): Path to save validation results.
    """

    train_results_path.mkdir(parents=True, exist_ok=True)
    val_results_path.mkdir(parents=True, exist_ok=True)

    # Paths for training models
    yolo11n_model_path = 'yolo11n.pt'
    custom_model_path = yamls_path / 'custom_model.yaml'

    # Paths for training configurations
    sdg_from_scratch = yamls_path / 'sgd_from_scratch.yaml'
    adamw_finetuning = yamls_path / 'adam_finetuning.yaml'
    adamw_transfer_learning = yamls_path / 'adamw_transfer_learning.yaml'
    

    # Path for .pt files
    model_finetuning_yaml_path = train_results_path / 'model_finetuning' / 'weights' / 'last.pt'
    model_transfer_learning_yaml_path = train_results_path / 'model_transfer_learning' / 'weights' / 'last.pt'
    model_from_scratch_yaml_path = train_results_path / 'model_from_scratch' / 'weights' / 'last.pt'

    # Path to model validation results
    model_finetuning_val_results_path = val_results_path / 'model_finetuning' 
    model_transfer_learning_val_results_path = val_results_path / 'model_transfer_learning' 
    model_from_scratch_val_results_path = val_results_path / 'model_from_scratch'

    # 1) From scratch (pesos aleatorios) - SGD
    # Solo config, sin pesos preentrenados 
    train_model(custom_model_path, data_yml_path, sdg_from_scratch, train_results_path, 'model_from_scratch')
    #visualize_model(model_transfer_learning_yaml_path, data_yml_path)

    # 2) Transfer learning (modelo preentrenado YOLO11n, congelar backbone y neck) - AdamW
    train_model(yolo11n_model_path, data_yml_path, adamw_finetuning, train_results_path, 'model_transfer_learning', unfreeze=2)
    #visualize_model(model_finetuning_yaml_path, data_yml_path)
    #("./training_results/yolo11n_model/weights/last.pt")

    # 3) Fine-tuning (modelo preentrenado YOLO11n, entrenar todo) - AdamW
    # No congelar nada, entrenar todo 
    train_model(yolo11n_model_path, data_yml_path, adamw_transfer_learning, train_results_path, 'model_finetuning')
    #evaluate_model(model_finetuning_yaml_path, model_finetuning_val_results_path, data_yml_path)



if __name__ == "__main__":

    # Paths
    main_path = Path(__file__).parent.parent
    dataset_path: Path = main_path / 'dataset'
    yamls_path: Path =  main_path / "yamls"

    # Paths for images and labels
    labels_path: Path = dataset_path / 'labels'
    images_path: Path = dataset_path / 'images'

    # Paths for COCO annotations
    # Original COCO JSON file and the one to be converted to YOLO format
    original_coco_json_file: Path = labels_path / 'original_coco_labels.json'
    coco_json_file: Path = labels_path / 'coco_labels.json'

    # Path where you want to save YOLO annotations
    yolo_dataset_path: Path =  dataset_path / 'yolo_dataset.yaml'

    original_names_map: Path = images_path / 'original_names_map.json'
    unprocessed_images_path: Path = images_path / 'unprocessed'

    # Paths for renamed images and labels
    renamed_images_path: Path = images_path / 'renamed'
    renamed_labels_path: Path = labels_path / 'renamed' 

    # Paths for train splits and augmented data
    train_images_path: Path =  images_path / 'train'
    train_labels_path: Path = labels_path / 'train'
    train_aug_images_path: Path = images_path / 'train_augmented'
    train_aug_labels_path: Path = labels_path / 'train_augmented'

    # Path for training results
    train_results_path = main_path / 'training_results'
    val_results_path = main_path / 'val_results'

    start_time: float = time.time()

    # Split the dataset, convert COCO annotations to YOLO format, and augment the training data
    # create_data(
    #     images_path, labels_path, renamed_images_path, renamed_labels_path,
    #     original_coco_json_file, coco_json_file, original_names_map, unprocessed_images_path, yolo_dataset_path, 
    #     train_images_path, train_labels_path, train_aug_images_path, train_aug_labels_path
    # )

    # Visualize the dataset
    # visualize_dataset(str(dataset_path), "yolo", "train") #, str(original_coco_json_file))

    # Train the model
    train_models(yamls_path, yolo_dataset_path, train_results_path, val_results_path)

    end_time: float = time.time() 
    elapsed: float = end_time - start_time
    print(f"Start: {start_time} - End: {end_time}")
    print(f"Total execution time: {elapsed:.4f} seconds")