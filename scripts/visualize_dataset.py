from argparse import ArgumentParser
import fiftyone as fo
import json
from pathlib import Path
import re
from progress_bar import printProgressBar

def  load_coco_dataset(dataset_dir):
    # Load COCO formatted dataset
    coco_dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path= f"{dataset_dir}/images",
        labels_path= f"{dataset_dir}/labels_coco.json",
        include_id=True,
    )
    coco_dataset.compute_metadata()
    return coco_dataset

def load_yolo_dataset(dataset_dir, split):
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.YOLOv5Dataset, 
        dataset_dir= dataset_dir, 
        yaml_path=f"{dataset_dir}/dataset_yolo.yaml", 
        split=split
    )
    return dataset

def add_original_names_field(dataset: fo.Dataset, dataset_dir: str):
    json_path = Path(dataset_dir) / "original_names_map.json"

    with open(json_path) as f:
        original_to_new = json.load(f)

    # Crear mapa inverso en memoria (nuevo -> original)
    new_to_original = {v: k for k, v in original_to_new.items()}

    total = len(dataset)
    printProgressBar(0, total, prefix='Adding original names:', suffix='Complete', length=50)

    # Añadir campo a cada sample
    for idx, sample in enumerate(dataset, start=1):
        filename = Path(sample.filepath).name

        # Eliminar el sufijo _augX si existe (ej: 00001_aug1.jpg → 00001.jpg)
        match = re.match(r"^(.*)_aug\d+\.(\w+)$", filename)
        base_name = f"{match.group(1)}.{match.group(2)}" if match else filename

        # Buscar nombre original en el mapa
        original = new_to_original.get(base_name, "desconocido")

        # Guardar el campo en el sample
        sample["original_name"] = original
        sample.save()

        if (idx * 100 // total) != ((idx - 1) * 100 // total):
            printProgressBar(idx, total, prefix='Adding original names:', suffix='Complete', length=50)


def arguments():
    parser=ArgumentParser(description="Visualize NN/CV project dataset")
    parser.add_argument("--path","--p",type=str,help="Path to the root folder of the dataset")
    parser.add_argument("--format","--f",type=str,choices=["yolo","coco"],help="Format of the dataset")
    parser.add_argument("--split","--s",required=False,type=str,choices=["train","val","test"],help="Split of the dataset (only for YOLO format)")
    parser.add_argument("--names","--n",required=False,type=str,help="Original names CSV file (only for YOLO format)")
    args=parser.parse_args()
    return args

def visualize_dataset(args):
    if args.format =="coco":
        dataset = load_coco_dataset(args.path)
    elif args.format =="yolo":
        dataset = load_yolo_dataset(args.path, args.split)
    else:
        raise ValueError("Unsupported dataset format")
    
    # Add original names field if the dataset is in YOLO format 
    add_original_names_field(dataset, args.names)
    
    session = fo.launch_app(dataset)
    session.wait()

if __name__ == "__main__":
    
    args = arguments()
    visualize_dataset(args)