# coco_converter.py
import json
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple
from tqdm import tqdm
from utils.config_logging import logger


class CocoConverter:
    """
    A class responsible for converting COCO annotations to the 
    YOLO format and managing the related dataset configuration files (YAML).
    """

    @staticmethod
    def _get_category_maps(coco_data: Dict[str, Any]) -> Tuple[Dict[int, int], Dict[int, str]]:
        """
        Filters categories (excluding supercategories) and creates mappings 
        for category ID -> class index and class index -> class name.
        
        Returns:
            Tuple[Dict[int, int], Dict[int, str]]: category_id_to_class, class_index_to_name
        """

        # 1. Identify Supercategories (to exclude them as classes)
        supercategories: set[str] = {
            cat.get('supercategory') for cat in coco_data.get('categories', [])
            if cat.get('supercategory') and cat.get('supercategory').lower() != 'none'
        }

        # 2. Filter Categories (only use child categories as classes)
        filtered_categories: List[Dict[str, Any]] = [
            cat for cat in coco_data.get('categories', []) 
            if cat['name'] not in supercategories
        ]

        # Ensure consistent indexing by sorting by ID
        filtered_categories = sorted(filtered_categories, key=lambda x: x['id'])

        # 3. Create Mappings
        category_id_to_class: Dict[int, int] = {}
        class_index_to_name: Dict[int, str] = {}
        
        for idx, category in enumerate(filtered_categories):
            category_id_to_class[category['id']] = idx
            class_index_to_name[idx] = category['name']
            
        logger.info(f"Loaded {len(class_index_to_name)} final classes for conversion.")
        return category_id_to_class, class_index_to_name


    @classmethod
    def convert_annotations(
        cls,
        coco_data: Dict[str, Any],
        labels_path: Path
    ) -> Dict[int, str]:
        """
        Converts COCO annotations (bbox) to normalized YOLO format and saves 
        them to individual .txt files per image.
        
        Args:
            coco_data (Dict[str, Any]): Loaded COCO dataset dictionary.
            labels_path (Path): Directory where YOLO annotations will be saved.
            
        Returns:
            Dict[int, str]: Mapping of class index -> class name used for conversion.
        """
        
        category_id_to_class, class_index_to_name = cls._get_category_maps(coco_data)
        
        # Map image ID to (file_name, width, height)
        image_map: Dict[int, Tuple[str, int, int]] = {
            image['id']: (image['file_name'], image['width'], image['height']) 
            for image in coco_data.get('images', [])
        }

        # Accumulate annotations per image
        annotations_by_image: Dict[str, List[str]] = defaultdict(list)
        total_annotations: int = len(coco_data.get('annotations', []))
        
        logger.info(f"Processing {total_annotations} COCO annotations...")

        # --- PHASE 1: Process Annotations and Convert ---
        for ann in tqdm(
            coco_data.get('annotations', []), 
            desc='Converting annotations', 
            unit='ann', 
            ncols=100
        ):
            annotations_cat_id: int = ann['category_id']
            if annotations_cat_id not in category_id_to_class:
                continue # Skip annotations for filtered categories

            # BBox: [x_min, y_min, width, height]
            x_min: float = ann['bbox'][0]
            y_min: float = ann['bbox'][1]
            width: float = ann['bbox'][2]
            height: float = ann['bbox'][3]
            image_id: int = ann["image_id"]
            
            # Retrieve image dimensions
            try:
                image_name, image_width, image_height = image_map[image_id]
            except KeyError:
                # This should ideally not happen if data is clean
                logger.warning(f"Image ID {image_id} not found in image_map. Skipping annotation.")
                continue

            # Calculate normalized YOLO coordinates (x_center, y_center, width_norm, height_norm)
            x_center: float = (x_min + width / 2) / image_width
            y_center: float = (y_min + height / 2) / image_height
            width_norm: float = width / image_width
            height_norm: float = height / image_height
            
            class_id: int = category_id_to_class[annotations_cat_id]
            
            # YOLO format: class_id x_center y_center width_norm height_norm
            yolo_annotation: str = f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n"
            annotations_by_image[image_name].append(yolo_annotation)
        
        logger.info(f"Total images with annotations: {len(annotations_by_image)}")

        # --- PHASE 2: Write YOLO Annotation Files ---
        labels_path.mkdir(parents=True, exist_ok=True)
        
        for image_name, annotations in tqdm(
            annotations_by_image.items(), 
            desc='Writing YOLO files', 
            unit='file', 
            ncols=100
        ):
            # Use image stem (name without extension) for label file name
            output_file: Path = labels_path / (Path(image_name).stem + '.txt')
            with output_file.open('w') as f:
                f.writelines(annotations)

        logger.info(f"Conversion completed. YOLO annotations saved to {labels_path}")
        return class_index_to_name

    def get_filtered_class_names(coco_data: Dict[str, Any]) -> Dict[int, str]:
        """
        Retrieve the mapping of class indices to class names, excluding supercategories.
        
        This method uses the internal _get_category_maps function to filter out 
        supercategories and create a mapping suitable for YOLO conversion.
        
        Args:
            coco_data (Dict[str, Any]): COCO dataset loaded from JSON.
        
        Returns:
            Dict[int, str]: Dictionary mapping class index -> class name
        """
        _, names_dict = CocoConverter._get_category_maps(coco_data)
        return names_dict

    @staticmethod
    def create_yaml_config(class_names: Dict[int, str], yaml_output_path: Path) -> None:
        """
        Creates a standard YOLO dataset configuration YAML file.
        
        Args:
            class_names (Dict[int, str]): Dictionary mapping class indices (0, 1, ...) to class names.
            yaml_output_path (Path): Path where the YAML file will be saved.
        """
        data_yaml: Dict[str, Any] = {
            # Default paths for split directories relative to the dataset folder
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            # Class names ordered by index
            'names': list(class_names.values()) 
        }

        yaml_output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_output_path, 'w') as f:
            yaml.dump(data_yaml, f, sort_keys=False)

        logger.info(f"YOLO config YAML saved to: {yaml_output_path}")


    @staticmethod
    def update_train_path(yaml_path: Path, new_train_path: str) -> None:
        """
        Updates the 'train' key in a YOLO dataset YAML file, typically for augmentation.
        
        Args:
            yaml_path (Path): Path to the YAML file.
            new_train_path (str): New path (e.g., 'images/train_augmented').
        """
        with open(yaml_path, 'r') as f:
            data: Dict[str, Any] = yaml.safe_load(f)

        data['train'] = new_train_path

        with open(yaml_path, 'w') as f:
            yaml.dump(data, f)

        logger.info(f"Updated YAML train path to: {new_train_path}")


    @staticmethod
    def create_new_train_yaml(original_yaml_path: Path, new_yaml_path: Path, new_train_path: str) -> None:
        """
        Creates a new YOLO dataset YAML file based on an existing one, 
        but with a modified 'train' path.
        
        Args:
            original_yaml_path (Path): Path to the original YAML file.
            new_yaml_path (Path): Path where the new YAML file will be saved.
            new_train_path (str): New train path (e.g., 'images/train_augmented').
        """
        # Load the original YAML
        with open(original_yaml_path, 'r') as f:
            data: Dict[str, Any] = yaml.safe_load(f)

        # Update the train path
        data['train'] = new_train_path

        # Write to the new YAML file
        with open(new_yaml_path, 'w') as f:
            yaml.dump(data, f)

        logger.info(f"Created new YAML at {new_yaml_path} with train path: {new_train_path}")


    @staticmethod
    def update_coco_json_filenames(
        json_map_path: Path, 
        coco_json_path: Path, 
        output_json_path: Path
    ) -> None:
        """
        Loads a name map (original_name -> new_name), updates the 'file_name' field
        in the COCO JSON's 'images' section, and saves the modified JSON.
        
        Args:
            json_map_path (Path): Path to JSON file mapping original filenames to new filenames.
            coco_json_path (Path): Path to original COCO JSON file.
            output_json_path (Path): Path to save the updated COCO JSON.
        """
        
        # Load the JSON map original_name -> new_name
        with open(json_map_path, "r") as f:
            name_map: Dict[str, str] = json.load(f)

        # Load COCO JSON
        with open(coco_json_path, "r") as f:
            coco: Dict = json.load(f)

        # Replace file_name in images according to the map
        replaced_count: int = 0
        for img in coco.get("images", []):
            original_name: str = img["file_name"]
            if original_name in name_map:
                img["file_name"] = name_map[original_name]
                replaced_count += 1
            else:
                # Log a warning if a file name in COCO wasn't in the provided map
                # (Can happen if the map is incomplete or there are unused COCO entries)
                logger.warning(f"Image '{original_name}' not found in renaming map.")

        logger.info(f"Total image filenames renamed in COCO JSON: {replaced_count}")

        # Save modified JSON with indentation for readability
        with open(output_json_path, "w") as f:
            json.dump(coco, f, indent=4)
        
        logger.info(f"Updated COCO JSON saved to: {output_json_path}")