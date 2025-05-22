import json, time 
from pathlib import Path

from convert_coco_to_yolo import verify_path, convert_labels, create_yaml_from_coco, get_filtered_categories_and_names
from visualize_dataset import visualize_dataset, arguments
from split_dataset import map_images, strartified_data, save_images
from train_model import train


dataset_dir = Path.cwd() / 'dataset'


def convert_coco_to_yolo():
    # Path to the COCO annotations JSON file
    coco_json_file = dataset_dir / 'labels' / 'labels_coco.json'

    # Path where you want to save YOLO annotations
    output_dir = dataset_dir / 'labels' / 'unprocessed' 
    yaml_output_path = dataset_dir / 'yamls' / 'dataset_yolo.yaml'

    # Verify the output directory exists
    verify_path(output_dir)

    try:
        # Load COCO annotations
        with open(coco_json_file, 'r') as f:
            coco_data = json.load(f)

    except Exception as e:
        print(f"Error loading COCO JSON file: {e}")
        return

    filtered_categories, names = get_filtered_categories_and_names(coco_data)
    convert_labels(coco_data, filtered_categories, output_dir)
    create_yaml_from_coco(names, yaml_output_path)


def split_dataset():
    # Directorios
    source_images_dir = dataset_dir / 'images' / 'unprocessed' 
    source_labels_dir = dataset_dir / 'labels' / 'unprocessed' 

    if not source_images_dir.exists():
        raise Exception(f"Source images directory not found: {source_images_dir}")

    # Obtener lista de imágenes
    image_files = [f for ext in ('*.jpg', '*.jpeg', '*.png') for f in source_images_dir.glob(ext)]
    print(f"Found {len(image_files)} images")

    image_to_class = map_images(image_files, source_labels_dir)

    if not image_to_class:
       raise Exception("No images with annotations found in source directory.")

    # Dataset para split
    images = list(image_to_class.keys())
    train_images, val_images, test_images = strartified_data(images)

    # Asociar cada split con su correspondiente lista de imágenes
    split_data = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }

    for split, images in split_data.items():
        images_dir = dataset_dir / 'images' / split
        labels_dir = dataset_dir / 'labels' / split

        # Crear la carpeta 'images' y 'labels' si no existen
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        save_images(images, source_labels_dir, source_images_dir, labels_dir, images_dir, split)

    print(f"Total images: {len(images)}")

    for split, images in split_data.items():
        print(f"{split.capitalize()}: {len(images)}")


def create_data():
    start_time = time.time()
    convert_coco_to_yolo()
    split_dataset()
    end_time = time.time() 
    elapsed = end_time - start_time
    print(f"Tiempo total de ejecución: {elapsed:.4f} segundos")

if __name__ == "__main__":
    args = arguments()
    #visualize_dataset(args)

    #create_data()
    train()