import json
import shutil
from pathlib import Path
from progress_bar import printProgressBar

def rename_images(folder, renamed_folder, json_path):
    renamed_folder.mkdir(exist_ok=True)

    print(f"Sorting files in {folder}...")
    # consistent ordering
    files = sorted([
        f for f in folder.iterdir()
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ])
    print(f"Found {len(files)} images to rename.")

    name_map = {}
    total = len(files)
    printProgressBar(0, total, prefix='Renaming files:', suffix='Complete', length=50)

    for idx, file_path in enumerate(files, start=1):
        ext = file_path.suffix.lower()
        new_name = f"{idx:05d}{ext}"
        new_path = renamed_folder / new_name

        shutil.copy2(file_path, new_path) 
        name_map[file_path.name] = new_name
        # print(f"Rename: {file_path.name} → {new_name}")

        if (idx * 100 // total) != ((idx - 1) * 100 // total):
            printProgressBar(idx, total, prefix='Renaming files:', suffix='Complete', length=50)

    with open(json_path, 'w') as f:
        json.dump(name_map, f, indent=2)

    print(f"\nRenaming complete. Mapping saved to {json_path}")

def update_coco_json(json_map_path: Path, coco_json_path: Path, output_json_path: Path):
    # Cargar el mapa JSON original_name -> new_name
    with open(json_map_path, "r") as f:
        name_map = json.load(f)

    # Carga JSON COCO
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    # Reemplaza file_name en imágenes según el CSV
    replaced_count = 0
    for img in coco["images"]:
        original_name = img["file_name"]
        if original_name in name_map:
            img["file_name"] = name_map[original_name]
            replaced_count += 1
        else:
            print(f"WARNING: {original_name} no encontrado en JSON")

    print(f"Total imágenes renombradas en JSON: {replaced_count}")

    # Guarda JSON modificado
    with open(output_json_path, "w") as f:
        json.dump(coco, f, indent=4)