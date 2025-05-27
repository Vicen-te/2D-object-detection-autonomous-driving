# augmentation_yolo.py
from concurrent.futures import ThreadPoolExecutor, Future
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from pathlib import Path
import random
import math
from progress_bar import printProgressBar


def yolo_to_bbox(box, img_width, img_height):
    """
    Convierte bbox YOLO normalizado a (x_min, y_min, x_max, y_max) en pixeles.
    """
    x_c, y_c, w, h = box
    x_c *= img_width
    y_c *= img_height
    w *= img_width
    h *= img_height
    x_min = x_c - w / 2
    y_min = y_c - h / 2
    x_max = x_c + w / 2
    y_max = y_c + h / 2
    return (x_min, y_min, x_max, y_max)

def bbox_to_yolo(box, img_width, img_height):
    """
    Convierte bbox absoluto a formato YOLO normalizado.
    """
    x_min, y_min, x_max, y_max = box
    w = x_max - x_min
    h = y_max - y_min
    x_c = x_min + w / 2
    y_c = y_min + h / 2
    return (x_c / img_width, y_c / img_height, w / img_width, h / img_height)

def get_affine_matrix(angle_deg, translate, scale, center):
    """
    Devuelve la matriz de transformación affine (2x3) centrada como en torchvision.
    """
    
    angle = math.radians(angle_deg)
    alpha = scale * math.cos(angle)
    beta = scale * math.sin(angle)
    tx, ty = translate
    cx, cy = center

    # Ajustar para que la rotación/escala sea respecto al centro
    # Matriz: T * R * T^-1
    matrix = [
        [alpha, -beta, (1 - alpha) * cx + beta * cy + tx],
        [beta,  alpha, (1 - alpha) * cy - beta * cx + ty]
    ]
    return matrix

def apply_affine_to_point(x, y, matrix):
    """
    Aplica la transformación affine 2x3 a un punto (x,y).
    """
    new_x = matrix[0][0]*x + matrix[0][1]*y + matrix[0][2]
    new_y = matrix[1][0]*x + matrix[1][1]*y + matrix[1][2]
    return (new_x, new_y)

def transform_bounding_box(box, matrix):
    """
    Aplica la matriz affine a las 4 esquinas del bbox y devuelve bbox nuevo.
    """
    x_min, y_min, x_max, y_max = box
    corners = [
        (x_min, y_min),
        (x_min, y_max),
        (x_max, y_min),
        (x_max, y_max)
    ]
    transformed = [apply_affine_to_point(x, y, matrix) for x, y in corners]
    xs, ys = zip(*transformed)
    return (min(xs), min(ys), max(xs), max(ys))

def read_yolo_labels(label_path):
    labels = []
    if not label_path.exists():
        return labels  # No hay etiquetas
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                labels.append([class_id] + coords)
    return labels

def save_yolo_labels(labels, save_path):
    with open(save_path, 'w') as f:
        for label in labels:
            class_id = label[0]
            coords = label[1:]
            coords_str = " ".join(f"{c:.6f}" for c in coords)
            f.write(f"{class_id} {coords_str}\n")

def augment_image_with_matrix(image):
    angle = random.uniform(-30, 30)
    max_dx = 0.2 * image.width
    max_dy = 0.2 * image.height
    translate_x = random.uniform(-max_dx, max_dx)
    translate_y = random.uniform(-max_dy, max_dy)
    scale = random.uniform(0.8, 1.2)

    image_aug = transforms.functional.affine(
        image,
        angle=angle,
        translate=(int(translate_x), int(translate_y)),
        scale=scale,
        shear=0,
        interpolation=InterpolationMode.BILINEAR,
        fill=None
    )

    center = (image.width / 2.0, image.height / 2.0)
    matrix = get_affine_matrix(angle, (translate_x, translate_y), scale, center)
    return image_aug, matrix


def clamp_bbox(box, img_w, img_h):
    x_min, y_min, x_max, y_max = box
    x_min = max(0, min(x_min, img_w))
    y_min = max(0, min(y_min, img_h))
    x_max = max(0, min(x_max, img_w))
    y_max = max(0, min(y_max, img_h))
    return (x_min, y_min, x_max, y_max)


def is_valid_box(box, min_size=1):
    x_min, y_min, x_max, y_max = box
    w = x_max - x_min
    h = y_max - y_min
    return w >= min_size and h >= min_size


def transform_yolo_labels(labels, matrix, orig_size, new_size,):
    orig_w, orig_h = orig_size
    new_w, new_h = new_size

    new_labels = []
    for label in labels:
        class_id = label[0]
        x_c, y_c, w, h = label[1:]

        # Convertir de YOLO (centrado, normalizado) a bbox absoluta (xmin, ymin, xmax, ymax)
        box_abs = yolo_to_bbox((x_c, y_c, w, h), orig_w, orig_h)

        # Transformar la bbox usando la matriz affine
        box_trans = transform_bounding_box(box_abs, matrix)

        # Clamp para que la bbox esté dentro del nuevo tamaño de imagen
        box_clamped = clamp_bbox(box_trans, new_w, new_h)

        # Filtrar cajas inválidas o demasiado pequeñas
        if not is_valid_box(box_clamped):
            continue  # Ignorar esta caja

        # Convertir bbox absoluto a formato YOLO normalizado en la nueva imagen
        box_yolo = bbox_to_yolo(box_clamped, new_w, new_h)
        new_labels.append([class_id, *box_yolo])

    return new_labels 

def process_single_image(img_path, input_label_folder, output_img_folder, output_label_folder, augmentations_per_image):
    image = Image.open(img_path).convert("RGB")
    orig_size = (image.width, image.height)

    labels_path = input_label_folder / f"{img_path.stem}.txt"
    labels = read_yolo_labels(labels_path)

    for i in range(augmentations_per_image):
        aug_image, matrix = augment_image_with_matrix(image)
        new_size = (aug_image.width, aug_image.height)

        aug_labels = transform_yolo_labels(labels, matrix, orig_size, new_size)

        out_img_name = f"{img_path.stem}_aug{i+1}{img_path.suffix}"
        out_img_path = output_img_folder / out_img_name
        aug_image.save(out_img_path)

        out_label_name = f"{img_path.stem}_aug{i+1}.txt"
        out_label_path = output_label_folder / out_label_name
        save_yolo_labels(aug_labels, out_label_path)

        # print(f"Saved image: {out_img_path} and labels: {out_label_path}")

    # Guardar original
    orig_img_path = output_img_folder / f"{img_path.name}"
    image.save(orig_img_path)

    orig_label_path = output_label_folder / f"{img_path.stem}.txt"
    save_yolo_labels(labels, orig_label_path)

    # print(f"Copied original image: {orig_img_path} and labels: {orig_label_path}")

def augment_dataset(input_img_folder: Path, input_label_folder: Path, output_img_folder: Path, output_label_folder: Path, augmentations_per_image: int=3):
    output_img_folder.mkdir(parents=True, exist_ok=True)
    output_label_folder.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in input_img_folder.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    if not image_paths:
        print("No images found in", input_img_folder)
        return
    
    total = len(image_paths)
    print(f"Found {total} images in {input_img_folder}")

    with ThreadPoolExecutor() as executor:
        futures: list[Future] = []
        for img_path in image_paths: # for idx, img_path in enumerate(image_paths, start=1):
            futures.append(executor.submit(process_single_image, img_path, input_label_folder, output_img_folder, output_label_folder, augmentations_per_image))

        try:
            print("Starting augmentation...")
            total = len(futures)
            printProgressBar(0, total, prefix='Augmenting images:', suffix='Complete', length=50)
            for idx, future in enumerate(futures, start=1):
                future.result()  # Esperar a que termine, lanzar excepción si hay error
                printProgressBar(idx, total, prefix='Augmenting images:', suffix='Complete', length=50)
                
        except KeyboardInterrupt:
            print("\nInterrupción detectada. Cancelando tareas...")
            for future in futures:
                future.cancel()  # intenta cancelar las tareas pendientes
            executor.shutdown(wait=False)
            print("Tareas canceladas, saliendo...")
            raise  # para terminar el programa

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data augmentation for images and YOLO labels")
    parser.add_argument("--input_img_folder", type=str, required=True, help="Carpeta con imágenes originales")
    parser.add_argument("--input_label_folder", type=str, required=True, help="Carpeta con etiquetas YOLO originales")
    parser.add_argument("--output_img_folder", type=str, required=True, help="Carpeta para imágenes aumentadas")
    parser.add_argument("--output_label_folder", type=str, required=True, help="Carpeta para etiquetas aumentadas")
    parser.add_argument("--num_aug", type=int, default=3, help="Número de augmentaciones por imagen")

    args = parser.parse_args()

    augment_dataset(
        args.input_img_folder,
        args.input_label_folder,
        args.output_img_folder,
        args.output_label_folder,
        args.num_aug
    )
