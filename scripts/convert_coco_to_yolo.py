import yaml
from pathlib import Path
from progress_bar import printProgressBar
from collections import defaultdict

def verify_path(output_dir):

    if not output_dir.exists() or not output_dir.is_dir():
        print(f"Directory {output_dir} does not exist or is not a directory.")
        return
    
    files_iter = list(output_dir.iterdir())  # Necesitamos lista para conocer longitud y poder mostrar progreso
    total = len(files_iter)
    print(f"Found {total} files in {output_dir}")

    if total == 0:
        print("No files to delete.")
        return
    
    printProgressBar(0, total, prefix='Deleting old files:', suffix='Complete', length=50)

    for i, file in enumerate(files_iter, start=1):
        try:
            if file.is_file() or file.is_symlink():
                file.unlink()

            elif file.is_dir():
                # Solo elimina directorios vacíos para evitar errores
                try:
                    file.rmdir()
                except OSError:
                    pass  # Directorio no vacío, ignorar o agregar manejo si quieres borrar recursivamente

        except Exception as e:
            print(f"Error deleting {file}: {e}")

        # Actualiza barra solo si porcentaje visible cambia (ejemplo simple):
        if (i * 100 // total) != ((i - 1) * 100 // total):
            printProgressBar(i, total, prefix='Deleting old files:', suffix='Complete', length=50)

    print(f"\nDeleted all the existing files in folder: {output_dir}")

def convert_labels(coco_data, filtered_categories, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to map category IDs to class indices
    category_id_to_class = {
        category['id']: idx
        for idx, category in enumerate(filtered_categories)
    }

    # Dictionary to map images
    image_map = {
        image['id'] : [image['file_name'], image['width'], image['height']] 
        for image in coco_data['images']
    }

    # Dictionary to map images to their annotations
    annotations_by_image = defaultdict(list)

    total = len(coco_data['annotations'])
    
    # Initial call to print 0% progress
    printProgressBar(0, total, prefix='Saving Annotations:', suffix='Complete', length=50)

    for i, ann in enumerate(coco_data['annotations'], start=1):

        annotations_cat_id = ann['category_id']
        if annotations_cat_id not in category_id_to_class:
            # Omitir anotaciones cuya categoría no está en filtered_categories
            continue  

        # Get the bounding box details
        x_min, y_min, width, height = ann['bbox']

        # Get the image ID
        image_id = ann["image_id"]

        # Get the image name and dimensions
        image_name, image_width, image_height = image_map[image_id]
        
        # Normalize the bounding box
        x_center = (x_min + width / 2) / image_width
        y_center = (y_min + height / 2) / image_height
        width_norm  = width / image_width
        height_norm  = height / image_height
        
        # Get the class ID (YOLO uses class indices, not category names)
        class_id = category_id_to_class[annotations_cat_id]
        
        # Append the YOLO formatted annotation (class_id x_center y_center width height)
        yolo_annotation = f"{class_id} {x_center} {y_center} {width_norm} {height_norm}\n"
        annotations_by_image[image_name].append(yolo_annotation)

        # Update Progress Bar
        if (i * 100 // total) != ((i - 1) * 100 // total):
            printProgressBar(i, total, prefix = 'Saving Annotations:', suffix = 'Complete', length = 50)
    
    print(f"Total annotations: {total}")
    total = len(annotations_by_image)

    # Initial call to print 0% progress
    printProgressBar(0, total, prefix='Converting:', suffix='Complete', length=50)

    # Ahora escribimos archivos una vez por imagen
    for i, (image_name, annotations) in enumerate(annotations_by_image.items(), 1):
        output_file = output_dir / (Path(image_name).stem + '.txt')
        with output_file.open('w') as f:
            f.writelines(annotations)
        
        # Update Progress Bar
        if (i * 100 // total) != ((i - 1) * 100 // total):
            printProgressBar(i, total, prefix='Converting:', suffix='Complete', length=50)

    print(f"Total annotations by image: {total}")
    print(f"Conversion completed. YOLO annotations are saved in {output_dir}")

def get_filtered_categories_and_names(coco_data):
    # Obtener todos los nombres de supercategorías válidas 
    supercategories = set(cat.get('supercategory') for cat in coco_data['categories'] if cat.get('supercategory') and cat.get('supercategory').lower() != 'none')

    # Filtrar categorías que no son supercategorías (excluyendo las que están en supercategories)
    filtered_categories = [cat for cat in coco_data['categories'] if cat['name'] not in supercategories]

    # Ordenar filtradas por id para asegurar orden
    filtered_categories = sorted(filtered_categories, key=lambda x: x['id'])

    # Construir dict con índices secuenciales desde 0 y sus nombres
    names = {i: cat['name'] for i, cat in enumerate(filtered_categories)}

    return filtered_categories, names

def create_yaml_from_coco(names, yaml_output_path):

    # Crear diccionario final
    data_yaml = {
        'path': '../dataset/',
        'train': 'images/train',  # Relativo a 'path'
        'val': 'images/val',
        'test': 'images/test',  # opcional
        'names': names
    }

    yaml_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Guardar en YAML
    with open(yaml_output_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"Archivo YAML guardado en: {yaml_output_path}")