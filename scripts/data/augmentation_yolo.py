# augmentation_yolo.py
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Any
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from pathlib import Path
import random
import math
from tqdm import tqdm 
from utils.types_aliases import BBoxAbs, BBoxYolo, AffineMatrix

from utils.config_logging import setup_logging
logger = setup_logging()


class YoloAugmenter:
    """
    A professional class for performing geometric data augmentation on images
    and their corresponding YOLO-format bounding box annotations.
    """

    def __init__(self, aug_params: Dict[str, Any] = None):
        """
        Initializes the augmenter with augmentation parameters.
        
        Args:
            aug_params: Dictionary of parameters for random augmentation.
        """

        # Default parameters
        default_params = {
            'max_rotation_deg': 30,
            'max_translation_pct': 0.2,  # 20%
            'min_scale': 0.8,
            'max_scale': 1.2
        }

        # Merge user-provided params with defaults
        self.aug_params = {**default_params, **(aug_params or {})}

        # Validate parameters
        if self.aug_params['min_scale'] >= self.aug_params['max_scale']:
            raise ValueError("min_scale must be less than max_scale")

        if self.aug_params['max_rotation_deg'] < 0:
            raise ValueError("max_rotation_deg must be non-negative")


    @staticmethod
    def yolo_to_bbox(
        box: BBoxYolo,
        img_width: int,
        img_height: int
    ) -> BBoxAbs:
        """
        Convert normalized YOLO bbox (x_center, y_center, width, height)
        to absolute (x_min, y_min, x_max, y_max).

        Args:
            box: (x_center, y_center, width, height) normalized between 0 and 1
            img_width: Image width
            img_height: Image height
            
        Returns:
            Absolute bounding box as (x_min, y_min, x_max, y_max)
        """
        x_c, y_c, w, h = box
        
        # Scale to absolute coordinates
        x_c *= img_width
        y_c *= img_height
        w *= img_width
        h *= img_height
        
        # Calculate min/max
        x_min: float = x_c - w / 2
        y_min: float = y_c - h / 2
        x_max: float = x_c + w / 2
        y_max: float = y_c + h / 2

        return (x_min, y_min, x_max, y_max)


    @staticmethod
    def bbox_to_yolo(
        box: BBoxAbs,
        img_width: int,
        img_height: int
    ) -> BBoxYolo:
        """
        Convert absolute bbox coordinates (x_min, y_min, x_max, y_max) to normalized
        YOLO bbox format (x_center, y_center, width, height).
        
        Args:
            box: Absolute bounding box (x_min, y_min, x_max, y_max) in pixels
            img_width: Image width in pixels
            img_height: Image height in pixels
        
        Returns:
            Normalized bounding box as (x_center, y_center, width, height), values in [0, 1]
        """
        x_min, y_min, x_max, y_max = box

        w: float = x_max - x_min
        h: float = y_max - y_min
        x_c: float = x_min + w / 2
        y_c: float = y_min + h / 2

        # Normalize coordinates
        return (x_c / img_width, y_c / img_height, w / img_width, h / img_height)


    @staticmethod
    def _get_affine_matrix(
        angle_deg: float,
        translate: Tuple[float, float],
        scale: float,
        center: Tuple[float, float]
    ) -> AffineMatrix:
        """
        Compute the 2x3 affine transformation matrix centered on a given point.
        
        Args:
            angle_deg: Rotation angle in degrees (positive: counter-clockwise)
            translate: (tx, ty) translation in pixels
            scale: Scaling factor
            center: (cx, cy) coordinates for the center of rotation/scaling
        
        Returns:
            2x3 affine transformation matrix as a list of two lists (rows)
        """

        angle: float = math.radians(angle_deg)
        alpha: float = scale * math.cos(angle)
        beta: float = scale * math.sin(angle)

        tx, ty = translate
        cx, cy = center

        # Matrix for rotation/scaling around the center (cx, cy) and subsequent translation
        # Matrix: T * R * T^-1
        matrix: AffineMatrix = [
            [alpha, -beta, (1 - alpha) * cx + beta * cy + tx],
            [beta,   alpha, (1 - alpha) * cy - beta * cx + ty]
        ]

        return matrix


    @staticmethod
    def _apply_affine_to_point(
        x: float,
        y: float,
        matrix: AffineMatrix
    ) -> Tuple[float, float]:
        """
        Apply a 2x3 affine transformation matrix to a 2D point (x, y).
        
        Args:
            x: x coordinate of the point
            y: y coordinate of the point
            matrix: 2x3 affine transformation matrix
        
        Returns:
            Transformed point coordinates (x', y')
        """
        new_x: float = matrix[0][0]*x + matrix[0][1]*y + matrix[0][2]
        new_y: float = matrix[1][0]*x + matrix[1][1]*y + matrix[1][2]
        return (new_x, new_y)


    def _transform_bounding_box(
        self,
        box: BBoxAbs,
        matrix: AffineMatrix
    ) -> BBoxAbs:
        """
        Apply affine transformation to all 4 corners of a bounding box and
        return the new bounding box that contains the transformed points.
        
        Args:
            box: Bounding box (x_min, y_min, x_max, y_max)
            matrix: 2x3 affine transformation matrix
        
        Returns:
            Transformed bounding box (x_min, y_min, x_max, y_max)
        """
        x_min, y_min, x_max, y_max = box
        corners: List[Tuple[float, float]] = [
            (x_min, y_min),
            (x_min, y_max),
            (x_max, y_min),
            (x_max, y_max)
        ]
        
        # Transform all 4 corners
        transformed: List[Tuple[float, float]] = [
            self._apply_affine_to_point(x, y, matrix) for x, y in corners
        ]
        xs, ys = zip(*transformed)

        # Calculate the new minimum/maximum extent
        return (min(xs), min(ys), max(xs), max(ys))


    @staticmethod
    def read_yolo_labels(label_path: Path) -> List[List[float]]:
        """
        Read YOLO format labels from a text file.

        Args:
            label_path: Path to the label file

        Returns: 
            List of labels, each as [class_id, x_center, y_center, width, height]
        """
        labels: List[List[float]] = []
        if not label_path.exists():
            return labels
        
        with open(label_path, 'r') as f:
            for line in f:
                parts: List[str] = line.strip().split()
                if len(parts) == 5:
                    try:
                        class_id: int = int(parts[0])
                        coords: List[float] = list(map(float, parts[1:]))
                        labels.append([float(class_id)] + coords) # Use float for class_id to match list type
                    except ValueError:
                        logger.info(f"Warning: Skipping invalid line in {label_path}: {line.strip()}")
        return labels


    @staticmethod
    def save_yolo_labels(labels: List[List[float]], save_path: Path) -> None:
        """
        Save YOLO labels to a text file.
        
        Args:
            labels: List of labels [class_id, x_center, y_center, width, height]
            save_path: Path to the output file
        """
        with open(save_path, 'w') as f:
            for label in labels:
                class_id: int = int(label[0]) # Convert back to int for saving
                coords: List[float] = label[1:]
                # Format coordinates to 6 decimal places as is standard for YOLO
                coords_str: str = " ".join(f"{c:.6f}" for c in coords)
                f.write(f"{class_id} {coords_str}\n")


    def augment_image_with_matrix(self, image: Image.Image) -> Tuple[Image.Image, AffineMatrix]:
        """
        Perform random affine augmentation on an image and return the transformed image
        and its corresponding affine matrix.
        
        Random parameters:
        - rotation angle in [-30, 30] degrees
        - translation up to ±20% of image dimensions
        - scaling between 0.8 and 1.2
        
        Args:
            image: PIL Image to augment
        
        Returns:
            Tuple of (augmented_image, affine_matrix)
        """
        W, H = image.width, image.height
        params = self.aug_params

        # 1. Random Parameters Generation
        angle: float = random.uniform(-params['max_rotation_deg'], params['max_rotation_deg'])
        
        max_dx: float = params['max_translation_pct'] * W
        max_dy: float = params['max_translation_pct'] * H
        translate_x: float = random.uniform(-max_dx, max_dx)
        translate_y: float = random.uniform(-max_dy, max_dy)
        
        scale: float = random.uniform(params['min_scale'], params['max_scale'])

        # 2. Image Augmentation (using torchvision's affine transformation)
        image_aug: Image.Image = transforms.functional.affine(
            image,
            angle=angle,
            translate=(int(translate_x), int(translate_y)), # torchvision expects integers for translate
            scale=scale,
            shear=0,
            interpolation=InterpolationMode.BILINEAR,
            fill=None
        )

        # 3. Matrix Calculation (using floating point translation for BBox accuracy)
        center: Tuple[float, float] = (W / 2.0, H / 2.0)
        matrix: AffineMatrix = self._get_affine_matrix(angle, (translate_x, translate_y), scale, center)

        return image_aug, matrix


    @staticmethod
    def _clamp_bbox(box: BBoxAbs, img_w: int, img_h: int) -> BBoxAbs:
        """
        Clamp bounding box coordinates to lie within image boundaries.
        
        Args:
            box: Bounding box (x_min, y_min, x_max, y_max)
            img_w: Image width
            img_h: Image height
        
        Returns:
            Clamped bounding box coordinates
        """
        x_min, y_min, x_max, y_max = box
        x_min = max(0.0, min(x_min, img_w))
        y_min = max(0.0, min(y_min, img_h))
        x_max = max(0.0, min(x_max, img_w))
        y_max = max(0.0, min(y_max, img_h))
        return (x_min, y_min, x_max, y_max)


    @staticmethod
    def _is_valid_box(box: BBoxAbs, min_size: int = 1) -> bool:
        """
        Check if a bounding box is valid (width and height above min_size).
        
        Args:
            box: Bounding box (x_min, y_min, x_max, y_max)
            min_size: Minimum allowed width and height
        
        Returns:
            True if box is valid, False otherwise
        """
        x_min, y_min, x_max, y_max = box
        w: float = x_max - x_min
        h: float = y_max - y_min
        return w >= min_size and h >= min_size


    def transform_yolo_labels(
        self,
        labels: List[List[float]],
        matrix: AffineMatrix,
        orig_size: Tuple[int, int],
        new_size: Tuple[int, int]
    ) -> List[List[float]]:
        """
        Transform YOLO labels using an affine transformation matrix.
        
        Args:
            labels: List of YOLO labels [class_id, x_c, y_c, w, h]
            matrix: 2x3 affine transformation matrix
            orig_size: Original image size (width, height)
            new_size: New image size after transformation (width, height)
        
        Returns:
            List of transformed YOLO labels
        """
        orig_w, orig_h = orig_size
        new_w, new_h = new_size
        new_labels: List[List[float]] = []

        for label in labels:
            class_id: float = label[0]
            yolo_coords: BBoxYolo = tuple(label[1:]) # Extract (x_c, y_c, w, h)

            # 1. Convert YOLO (normalized) to absolute BBox (pixels)
            box_abs: BBoxAbs = self.yolo_to_bbox(yolo_coords, orig_w, orig_h)

            # 2. Transform BBox using the affine matrix
            box_trans: BBoxAbs = self._transform_bounding_box(box_abs, matrix)

            # 3. Clamp BBox to the new image boundaries
            box_clamped: BBoxAbs = self._clamp_bbox(box_trans, new_w, new_h)

            # 4. Filter invalid or too small boxes
            if not self._is_valid_box(box_clamped):
                continue

            # 5. Convert absolute BBox back to normalized YOLO format
            box_yolo: BBoxYolo = self.bbox_to_yolo(box_clamped, new_w, new_h)
            
            # Append as [class_id, x_c, y_c, w, h]
            new_labels.append([class_id, *box_yolo])

        return new_labels


    def _process_single_image(
        self,
        img_path: Path,
        input_labels_path: Path,
        output_images_path: Path,
        output_labels_path: Path,
        augmentations_per_image: int
    ) -> None:
        """
        Internal function to process a single image using augmentation:
        generate augmented copies and save them along with the original.

        Args:
            img_path: Path to the original image file
            input_labels_path: Path to the folder containing YOLO label files
            output_images_path: Path to the folder where augmented images will be saved
            output_labels_path: Path to the folder where augmented labels will be saved
            augmentations_per_image: Number of augmented copies to generate per image
        """
        try:
            image: Image.Image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.info(f"Error opening image {img_path}: {e}")
            return

        orig_size: Tuple[int, int] = (image.width, image.height)

        labels_path: Path = input_labels_path / f"{img_path.stem}.txt"
        labels: List[List[float]] = self.read_yolo_labels(labels_path)
        
        # Save augmented copies
        for i in range(augmentations_per_image):
            aug_image, matrix = self.augment_image_with_matrix(image)
            new_size: Tuple[int, int] = (aug_image.width, aug_image.height)

            aug_labels: List[List[float]] = self.transform_yolo_labels(labels, matrix, orig_size, new_size)

            aug_image_name: str = f"{img_path.stem}_aug{i+1}.jpg"
            aug_label_name: str = f"{img_path.stem}_aug{i+1}.txt"

            aug_image.save(output_images_path / aug_image_name)
            self.save_yolo_labels(aug_labels, output_labels_path / aug_label_name)

        # Save original image and labels unchanged in output folders (included in training data)
        image.save(output_images_path / img_path.name)
        self.save_yolo_labels(labels, output_labels_path / f"{img_path.stem}.txt")


    def augment_dataset(
        self,
        input_images_path: Path,
        input_labels_path: Path,
        output_images_path: Path,
        output_labels_path: Path,
        augmentations_per_image: int = 3,
        max_workers: int = 8
    ) -> None:
        """
        Augment an entire dataset using multithreading.
        
        Args:
            input_images_path: Path to the folder containing original images.
            input_labels_path: Path to the folder containing YOLO label files.
            output_images_path: Path to the folder where augmented images will be saved.
            output_labels_path: Path to the folder where augmented labels will be saved.
            augmentations_per_image: Number of augmentations to generate per image (including the original).
            max_workers: Maximum number of worker threads for parallel processing.
        """
        image_paths: List[Path] = sorted([
            p for p in input_images_path.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])
        
        if not image_paths:
            logger.info(f"No images found in {input_images_path}")
            return
        
        total_images: int = len(image_paths)
        logger.info(f"Found {total_images} images in {input_images_path}. Generating {augmentations_per_image + 1} total samples per image.")

        # Ensure output directories exist
        output_images_path.mkdir(parents=True, exist_ok=True)
        output_labels_path.mkdir(parents=True, exist_ok=True)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            # Submit tasks to the thread pool
            for img_path in image_paths:
                future = executor.submit(
                    self._process_single_image, 
                    img_path, 
                    input_labels_path, 
                    output_images_path, 
                    output_labels_path, 
                    augmentations_per_image
                )
                futures.append(future)

            # Use tqdm to show professional progress bar
            logger.info("Starting augmentation with multi-threading...")
            try:
                # Iterate over futures using tqdm to track progress
                for future in tqdm(
                    futures, 
                    desc="Processing images", 
                    unit="image", 
                    ncols=100
                ):
                    future.result()  # Wait for completion and raise exceptions
            
            except KeyboardInterrupt:
                logger.info("\nInterruption detected. Cancelling tasks...")
                for future in futures:
                    future.cancel()  # Attempt to cancel pending tasks
                executor.shutdown(wait=False)
                raise  # Exit program

        logger.info(f"Augmentation complete. Total images processed: {total_images}.")