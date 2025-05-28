from pathlib import Path
from progress_bar import printProgressBar
from sklearn.model_selection import train_test_split
import shutil
from collections import Counter

def map_images(image_files, source_labels_dir):

    total = len(image_files)

    # Initial call to print 0% progress
    printProgressBar(0, total, prefix = 'Checking Progress:', suffix = 'Complete', length = 50)

    # Mapear imagen a clases
    image_to_class = {}
    for i, image_file in enumerate(image_files, start=1):
        label_path = source_labels_dir / (image_file.stem + '.txt')
        
        if not label_path.exists():
            continue  # Saltar imágenes sin anotación

        with label_path.open('r') as f:
            classes = [int(line.split()[0]) for line in f if line.strip()]
            
            if classes:
                # Dominant-class stratification
                dominant_class = Counter(classes).most_common(1)[0][0] 
                image_to_class[image_file] = dominant_class

        # Actualizar barra cada vez que cambia el %
        if i * 100 // total != (i-1) * 100 // total:
            printProgressBar(i, total, prefix='Checking Progress:', suffix='Complete', length=50)

    return image_to_class

def strartified_data(images):

    # Primero split en train y temp (val + test)
    train_imgs, temp_imgs = train_test_split(images, test_size=0.2, random_state=42)

    # Luego split del temp en val y test (50% cada uno de 20% = 10% y 10%)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    # Split estratificado (train_images, val_images, test_imgs)
    return train_imgs, val_imgs, test_imgs
    

def save_images(images, source_labels_dir, source_images_dir, labels_dir, images_dir, text):
    
    total = len(images)

    # Initial call to print 0% progress
    printProgressBar(0, total, prefix = f'{text.capitalize()} Saving Progress:', suffix = 'Complete', length = 50)

    for i, image_file in enumerate(images, start=1):
        label_file = image_file.stem + '.txt'
        source_label_path = source_labels_dir / label_file

        if not source_label_path.exists():
            continue  # Saltar imágenes sin anotación

        source_image_path = source_images_dir / image_file.name
        destination_label_path = labels_dir / label_file
        destination_image_path = images_dir / image_file.name

        # Copiar contenido de etiqueta
        content = source_label_path.read_text()
        destination_label_path.write_text(content)

        # Copiar imagen con shutil para más rapidez y menos carga
        shutil.copy2(source_image_path, destination_image_path)

        # Actualizar barra cada vez que cambia el %
        if i * 100 // total != (i-1) * 100 // total:
            printProgressBar(i, total, prefix = f'{text.capitalize()} Saving Progress:', suffix = 'Complete', length=50)