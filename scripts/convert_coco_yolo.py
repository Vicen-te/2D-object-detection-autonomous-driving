import json
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple
from progress_bar import printProgressBar


def convert_labels(
    coco_data: Dict[str, Any],
    filtered_categories: List[Dict[str, Any]],
    labels_path: Path
) -> None:
    """
    Convert COCO annotations to YOLO format and save them in the specified directory.
    Args:
        coco_data (Dict[str, Any]): COCO dataset loaded from JSON.
        filtered_categories (List[Dict[str, Any]]): List of categories to include in the conversion.
        labels_path (Path): Directory where YOLO annotations will be saved.
    """
    
    # Dictionary to map category IDs to class indices
    category_id_to_class: Dict[int, int] = {
        category['id']: idx
        for idx, category in enumerate(filtered_categories)
    }

    # Dictionary to map images
    image_map: Dict[int, Tuple[str, int, int]] = {
        image['id'] : [image['file_name'], image['width'], image['height']] 
        for image in coco_data['images']
    }

    # Dictionary to map images to their annotations
    annotations_by_image: Dict[str, List[str]] = defaultdict(list)
    total: int = len(coco_data['annotations'])
    
    # Initial call to print 0% progress
    printProgressBar(0, total, prefix='Saving Annotations:', suffix='Complete', length=50)

    for i, ann in enumerate(coco_data['annotations'], start=1):
        annotations_cat_id: int = ann['category_id']
        if annotations_cat_id not in category_id_to_class:
            continue #< Skip annotations not in filtered categories

        # Get the bounding box details
        x_min: float = ann['bbox'][0]
        y_min: float = ann['bbox'][1]
        width: float = ann['bbox'][2]
        height: float = ann['bbox'][3]

        # Get the image ID
        image_id: int = ann["image_id"]

        # Get the image name and dimensions
        image_name: str = image_map[image_id][0]
        image_width: int = image_map[image_id][1]
        image_height: int = image_map[image_id][2]
        
        # Normalize the bounding box
        x_center: float = (x_min + width / 2) / image_width
        y_center: float = (y_min + height / 2) / image_height
        width_norm: float  = width / image_width
        height_norm: float  = height / image_height
        
        # Get the class ID (YOLO uses class indices, not category names)
        class_id: int = category_id_to_class[annotations_cat_id]
        
        # Append the YOLO formatted annotation (class_id x_center y_center width height)
        yolo_annotation: str = f"{class_id} {x_center} {y_center} {width_norm} {height_norm}\n"
        annotations_by_image[image_name].append(yolo_annotation)

        # Update Progress Bar
        if (i * 100 // total) != ((i - 1) * 100 // total):
            printProgressBar(i, total, prefix = 'Saving Annotations:', suffix = 'Complete', length = 50)
    
    print(f"Total annotations: {total}")
    total: int = len(annotations_by_image)

    # Initial call to print 0% progress
    printProgressBar(0, total, prefix='Converting:', suffix='Complete', length=50)

    # Ahora escribimos archivos una vez por imagen
    for i, (image_name, annotations) in enumerate(annotations_by_image.items(), 1):
        output_file: Path = labels_path / (Path(image_name).stem + '.txt')
        with output_file.open('w') as f:
            f.writelines(annotations)
        
        # Update Progress Bar
        if (i * 100 // total) != ((i - 1) * 100 // total):
            printProgressBar(i, total, prefix='Converting:', suffix='Complete', length=50)

    print(f"Total annotations by image: {total}")
    print(f"Conversion completed. YOLO annotations are saved in {labels_path}")



def get_filtered_categories_and_names(coco_data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[int, str]]:
    """
    Filter COCO categories to exclude supercategories and return a list of valid categories
    and a mapping of class indices to class names.
    Args:
        coco_data (Dict[str, Any]): COCO dataset loaded from JSON.
    Returns:
        Tuple[List[Dict[str, Any]], Dict[int, str]]: A tuple containing:
            - List of filtered categories (excluding supercategories).
            - Dictionary mapping class indices to class names.
    """

    # Obtener todos los nombres de supercategorías válidas 
    supercategories: set[str] = {
        cat.get('supercategory') for cat in coco_data['categories'] 
        if cat.get('supercategory') and cat.get('supercategory').lower() != 'none'
    }

    # Filtrar categorías que no son supercategorías (excluyendo las que están en supercategories)
    filtered_categories: List[Dict[str, Any]] = [
        cat for cat in coco_data['categories'] 
        if cat['name'] not in supercategories
    ]

    # Ordenar filtradas por id para asegurar orden
    filtered_categories: List[Dict[str, Any]] = sorted(filtered_categories, key=lambda x: x['id'])

    # Construir dict con índices secuenciales desde 0 y sus nombres
    names: Dict[int, str] = {i: cat['name'] for i, cat in enumerate(filtered_categories)}

    return filtered_categories, names



def create_yaml_from_coco(class_names: Dict[int, str], yaml_output_path: Path) -> None:
    """
    Create a YOLO dataset YAML file from COCO categories.
    Args:
        class_names  (Dict[int, str]): Dictionary mapping class indices to class names.
        yaml_output_path (Path): Path where the YAML file will be saved.
    """

    # Crear diccionario final
    data_yaml: Dict[str, Any] = {
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': class_names 
    }

    yaml_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Guardar en YAML
    with open(yaml_output_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"Archivo YAML guardado en: {yaml_output_path}")



def update_train_path(yaml_path: Path, new_train_path: str) -> None:
    """
    Update the 'train' path in a YOLO dataset YAML file.
    Args:
        yaml_path (Path): Path to the YAML file.
        new_train_path (str): New path for the training images.
    """

    data: Dict[str, Any]
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    data['train'] = new_train_path

    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)

    print(f"Updated train path to: {new_train_path}")


def update_coco_json(
    json_map_path: Path, 
    coco_json_path: Path, 
    output_json_path: Path
) -> None:
    """
    Load JSON map of original_name -> new_name, update COCO JSON filenames accordingly,
    and save the updated COCO JSON.
    Args:
        json_map_path (Path): Path to JSON file mapping original filenames to new filenames.
        coco_json_path (Path): Path to original COCO JSON file.
        output_json_path (Path): Path to save the updated COCO JSON.
    """
    
    # Load the JSON map original_name -> new_name
    with open(json_map_path, "r") as f:
        name_map: Dict = json.load(f)

    # Load COCO JSON
    with open(coco_json_path, "r") as f:
        coco: Dict = json.load(f)

    # Replace file_name in images according to the map
    replaced_count: int = 0
    for img in coco["images"]:
        original_name: str = img["file_name"]
        if original_name in name_map:
            img["file_name"] = name_map[original_name]
            replaced_count += 1
        else:
            print(f"WARNING: {original_name} no encontrado en JSON")

    print(f"Total images renamed in JSON: {replaced_count}")

    # Save modified JSON
    with open(output_json_path, "w") as f:
        json.dump(coco, f, indent=4)