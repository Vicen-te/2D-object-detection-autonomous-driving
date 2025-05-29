# augmentation_yolo.py
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Tuple
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from pathlib import Path
import random
import math
from progress_bar import printProgressBar



def yolo_to_bbox(
    box: Tuple[float, float, float, float],
    img_width: int,
    img_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert normalized YOLO bbox format (x_center, y_center, width, height) to absolute bbox
    coordinates (x_min, y_min, x_max, y_max) in pixels.
    
    Args:
        box: (x_center, y_center, width, height) normalized between 0 and 1
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Absolute bounding box as (x_min, y_min, x_max, y_max) in pixels
    """

    x_c: float
    y_c: float
    w: float
    h: float

    x_c, y_c, w, h = box
    x_c *= img_width
    y_c *= img_height
    w *= img_width
    h *= img_height
    
    x_min: float = x_c - w / 2
    y_min: float = y_c - h / 2
    x_max: float = x_c + w / 2
    y_max: float = y_c + h / 2

    return (x_min, y_min, x_max, y_max)



def bbox_to_yolo(
    box: Tuple[float, float, float, float],
    img_width: int,
    img_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert absolute bbox coordinates (x_min, y_min, x_max, y_max) in pixels to normalized
    YOLO bbox format (x_center, y_center, width, height).
    
    Args:
        box: Absolute bounding box (x_min, y_min, x_max, y_max) in pixels
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        Normalized bounding box as (x_center, y_center, width, height), values in [0, 1]
    """
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    x_min, y_min, x_max, y_max = box

    w: float = x_max - x_min
    h: float = y_max - y_min
    x_c: float = x_min + w / 2
    y_c : float= y_min + h / 2

    return (x_c / img_width, y_c / img_height, w / img_width, h / img_height)



def get_affine_matrix(
    angle_deg: float,
    translate: Tuple[float, float],
    scale: float,
    center: Tuple[float, float]
) -> List[List[float]]:
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

    tx: float
    ty: float
    cx: float
    cy: float

    tx, ty = translate
    cx, cy = center

    # Ajustar para que la rotación/escala sea respecto al centro
    # Matriz: T * R * T^-1
    matrix: List[List[float]] = [
        [alpha, -beta, (1 - alpha) * cx + beta * cy + tx],
        [beta,  alpha, (1 - alpha) * cy - beta * cx + ty]
    ]

    return matrix



def apply_affine_to_point(
    x: float,
    y: float,
    matrix: List[List[float]]
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



def transform_bounding_box(
    box: Tuple[float, float, float, float],
    matrix: List[List[float]]
) -> Tuple[float, float, float, float]:
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
    
    transformed: List[Tuple[float, float]] = [apply_affine_to_point(x, y, matrix) for x, y in corners]
    xs, ys = zip(*transformed)

    return (min(xs), min(ys), max(xs), max(ys))



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
        return labels  # No labels found
    
    with open(label_path, 'r') as f:
        for line in f:
            parts: List[str] = line.strip().split()

            # A valid YOLO label line should have 5 parts: class_id, x_center, y_center, width, height
            if len(parts) == 5:
                class_id: int = int(parts[0])
                coords: List[float] = list(map(float, parts[1:]))
                labels.append([class_id] + coords)

    return labels



def save_yolo_labels(labels: List[List[float]], save_path: Path) -> None:
    """
    Save YOLO labels to a text file.
    
    Args:
        labels: List of labels [class_id, x_center, y_center, width, height]
        save_path: Path to the output file
    """
    with open(save_path, 'w') as f:
        for label in labels:
            class_id: int  = label[0]
            coords: List[float] = label[1:]
            coords_str: str = " ".join(f"{c:.6f}" for c in coords)
            f.write(f"{class_id} {coords_str}\n")



def augment_image_with_matrix(image: Image.Image) -> Tuple[Image.Image, List[List[float]]]:
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

    angle: float = random.uniform(-30, 30)
    max_dx: float = 0.2 * image.width
    max_dy: float = 0.2 * image.height
    translate_x: float = random.uniform(-max_dx, max_dx)
    translate_y: float = random.uniform(-max_dy, max_dy)
    scale: float = random.uniform(0.8, 1.2)

    image_aug = transforms.functional.affine(
        image,
        angle=angle,
        translate=(int(translate_x), int(translate_y)),
        scale=scale,
        shear=0,
        interpolation=InterpolationMode.BILINEAR,
        fill=None
    )

    center: Tuple[float, float] = (image.width / 2.0, image.height / 2.0)
    matrix: List[List[float]] = get_affine_matrix(angle, (translate_x, translate_y), scale, center)

    return image_aug, matrix



def clamp_bbox(
    box: Tuple[float, float, float, float],
    img_w: int,
    img_h: int
) -> Tuple[float, float, float, float]:
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
    x_min = max(0, min(x_min, img_w))
    y_min = max(0, min(y_min, img_h))
    x_max = max(0, min(x_max, img_w))
    y_max = max(0, min(y_max, img_h))
    return (x_min, y_min, x_max, y_max)



def is_valid_box(box: Tuple[float, float, float, float], min_size: int = 1) -> bool:
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
    labels: List[List[float]],
    matrix: List[List[float]],
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
        class_id: int = label[0]
        x_c, y_c, w, h = label[1:]

        # Convertir de YOLO (centrado, normalizado) a bbox absoluta (xmin, ymin, xmax, ymax)
        box_abs: Tuple[float, float, float, float] = yolo_to_bbox((x_c, y_c, w, h), orig_w, orig_h)

        # Transformar la bbox usando la matriz affine
        box_trans: Tuple[float, float, float, float] = transform_bounding_box(box_abs, matrix)

        # Clamp para que la bbox esté dentro del nuevo tamaño de imagen
        box_clamped: Tuple[float, float, float, float] = clamp_bbox(box_trans, new_w, new_h)

        # Filtrar cajas inválidas o demasiado pequeñas
        if not is_valid_box(box_clamped):
            continue  # Ignorar esta caja

        # Convertir bbox absoluto a formato YOLO normalizado en la nueva imagen
        box_yolo: Tuple[float, float, float, float] = bbox_to_yolo(box_clamped, new_w, new_h)
        new_labels.append([class_id, *box_yolo])

    return new_labels 



def process_single_image(
    images_path: Path,
    input_labels_path: Path,
    output_images_path: Path,
    output_labels_path: Path,
    augmentations_per_image: int
) -> None:
    """
    Process a single image: perform augmentations, save augmented images and labels,
    and also save the original image and labels to output folders.

    Args:
        img_path: Path to the original image file
        input_labels_path: Path to the folder containing YOLO label files
        output_images_path: Path to the folder where augmented images will be saved
        output_labels_path: Path to the folder where augmented labels will be saved
        augmentations_per_image: Number of augmented copies to generate per image
    """
    image: Image.Image = Image.open(images_path).convert("RGB")
    orig_size: Tuple[int, int] = (image.width, image.height)

    labels_path: Path = input_labels_path / f"{images_path.stem}.txt"
    labels: List[List[float]] = read_yolo_labels(labels_path)

    for i in range(augmentations_per_image):
        aug_image, matrix = augment_image_with_matrix(image)
        new_size: Tuple[int, int] = (aug_image.width, aug_image.height)

        aug_labels: List[List[float]] = transform_yolo_labels(labels, matrix, orig_size, new_size)

        # Save augmented image and labels
        aug_image_name: str = f"{images_path.stem}_aug{i+1}.jpg"
        aug_label_name: str = f"{images_path.stem}_aug{i+1}.txt"

        aug_image.save(output_images_path / aug_image_name)
        save_yolo_labels(aug_labels, output_labels_path / aug_label_name)

        # print(f"Saved image: {aug_image_name} and labels: {aug_label_name}")

    # Save original image and labels unchanged in output folders
    image.save(output_images_path / images_path.name)
    save_yolo_labels(labels, output_labels_path / f"{images_path.stem}.txt")

    # print(f"Copied original image: {orig_img_path} and labels: {orig_label_path}")



def augment_dataset(
    input_images_path: Path,
    input_labels_path: Path,
    output_images_path: Path,
    output_labels_path: Path,
    augmentations_per_image: int = 3,
    max_workers: int = 8
) -> None:
    """
    Augment an entire dataset of images and their corresponding YOLO labels.
    Uses multithreading to speed up processing.

    Args:
        input_images_path: Path to the folder containing original images
        input_labels_path: Path to the folder containing YOLO label files
        output_images_path: Path to the folder where augmented images will be saved
        output_labels_path: Path to the folder where augmented labels will be saved
        augmentations_per_image: Number of augmentations to generate per image
        max_workers: Maximum number of worker threads to use for parallel processing
    """

    image_paths: List[Path] = sorted([p for p in input_images_path.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    if not image_paths:
        print("No images found in", input_images_path)
        return
    
    total: int = len(image_paths)
    print(f"Found {total} images in {input_images_path}")

    with ThreadPoolExecutor(max_workers) as executor:
        futures: List[Future] = []
        for img_path in image_paths: # for idx, img_path in enumerate(image_paths, start=1):
            futures.append(executor.submit(process_single_image, img_path, input_labels_path, output_images_path, output_labels_path, augmentations_per_image))

        try:
            print("Starting augmentation...")
            total: int = len(futures)
            printProgressBar(0, total, prefix='Augmenting images:', suffix='Complete', length=50)
            for idx, future in enumerate(futures, start=1):
                future.result()   # Wait for completion and raise exceptions if any
                printProgressBar(idx, total, prefix='Augmenting images:', suffix='Complete', length=50)
                
        except KeyboardInterrupt:
            print("\nInterrupción detectada. Cancelando tareas...")
            for future in futures:
                future.cancel()  # Attempt to cancel pending tasks
            executor.shutdown(wait=False)
            print("Tasks cancelled, exiting...")
            raise  # Exit program