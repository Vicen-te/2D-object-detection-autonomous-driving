import json, time 
from pathlib import Path

from convert_coco_to_yolo import verify_path, convert_labels, create_yaml_from_coco, get_filtered_categories_and_names, update_train_path
from visualize_dataset import visualize_dataset, arguments
from split_dataset import map_images, strartified_data, save_images
from train_model import train
from rename_dataset_images import rename_images, update_coco_json
from augmentation_yolo import augment_dataset


DATASET_DIR = Path.cwd() / 'dataset'

def convert_coco_to_yolo():
    # Path to the COCO annotations JSON file
    original_coco_json_file = DATASET_DIR / 'labels' / 'labels_coco_original.json'
    coco_json_file = DATASET_DIR / 'labels' / 'labels_coco.json'

    # Path where you want to save YOLO annotations
    output_dir = DATASET_DIR / 'labels' / 'renamed' 
    yaml_output_path =  Path(__file__).parent.parent / "yamls" / 'dataset_yolo.yaml'

    csv_path = DATASET_DIR / 'images' / 'original_names_map.json'
    source_images_folder = DATASET_DIR / 'images' / 'unprocessed'
    renamed_images_folder = DATASET_DIR / 'images' / 'renamed'

    rename_images(source_images_folder, renamed_images_folder, csv_path)
    update_coco_json(csv_path, original_coco_json_file, DATASET_DIR / 'labels' / 'labels_coco.json')

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
    # create_yaml_from_coco(names, yaml_output_path)


def split_dataset():
    # Directorios
    source_images_dir = DATASET_DIR / 'images' / 'renamed' 
    source_labels_dir = DATASET_DIR / 'labels' / 'renamed' 

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
        images_dir = DATASET_DIR / 'images' / split
        labels_dir = DATASET_DIR / 'labels' / split

        # Crear la carpeta 'images' y 'labels' si no existen
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        save_images(images, source_labels_dir, source_images_dir, labels_dir, images_dir, split)

    print(f"Total images: {len(images)}")

    for split, images in split_data.items():
        print(f"{split.capitalize()}: {len(images)}")

def train_augmented_dataset():
    input_img_folder =  DATASET_DIR / 'images' / 'train'
    input_label_folder = DATASET_DIR / 'labels' / 'train'

    output_img_folder = DATASET_DIR / 'images' / 'train_augmented'
    output_label_folder = DATASET_DIR / 'labels' / 'train_augmented'

    augment_dataset(input_img_folder, input_label_folder, output_img_folder, output_label_folder, augmentations_per_image=3)


def create_data():
    convert_coco_to_yolo()
    split_dataset()
    train_augmented_dataset()
    update_train_path(Path('yamls') / 'dataset_yolo.yaml',  'images/train_augmented')

if __name__ == "__main__":
    args = arguments()

    start_time = time.time()

    #create_data()
    #visualize_dataset(args)
    train()

    end_time = time.time() 
    elapsed = end_time - start_time
    print(f"comienzo{start_time} - fin{end_time}")
    print(f"Tiempo total de ejecución: {elapsed:.4f} segundos")

    