from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Optional
import fiftyone as fo
import json, re

from progress_bar import printProgressBar

def  load_coco_dataset(images_path: str, labels_path: str) -> fo.Dataset:
    """
    Load a COCO dataset from a directory containing images and a JSON file with annotations.
    Args:
        images_path (str): Path to the directory containing images.
        labels_path (str): Path to the JSON file with COCO annotations.
    Returns:
        fo.Dataset: The loaded FiftyOne dataset.
    """
    # Load COCO dataset with FiftyOne
    coco_dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path= images_path,
        labels_path= labels_path,
        include_id=True,
    )
    coco_dataset.compute_metadata()
    return coco_dataset



def load_yolo_dataset(dataset_path: str, yaml_path: str, split: str) -> fo.Dataset:
    """
    Load a YOLO dataset from a directory and YAML file.
    Args:
        dataset_path (str): Path to the directory containing the YOLO dataset.
        yaml_path (str): Path to the YAML file defining the dataset structure.
        split (str): The split of the dataset to load (e.g., 'train', 'val', 'test').
    Returns:
        fo.Dataset: The loaded FiftyOne dataset.
    """
    # Load YOLO dataset with FiftyOne
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.YOLOv5Dataset, 
        dataset_dir=dataset_path, 
        yaml_path=yaml_path, 
        split=split
    )
    return dataset



def add_original_names_field(dataset: fo.Dataset, json_path: str) -> None:
    """
    Add a field to the dataset samples with the original names of the images.
    Args:
        dataset (fo.Dataset): The FiftyOne dataset to modify.
        json_path (str): Path to the JSON file mapping original names to new names.
    """

    original_to_new: Dict[str, str]
    with open(json_path) as f:
        original_to_new = json.load(f)

    # Create inverse map in memory (new -> original)
    new_to_original: Dict[str, str] = {v: k for k, v in original_to_new.items()}

    total: int = len(dataset)
    print(f"Total samples in dataset: {total}")
    printProgressBar(0, total, prefix='Adding original names:', suffix='Complete', length=50)

    for idx, sample in enumerate(dataset, start=1):
        filename: str = Path(sample.filepath).name

        # Remove the _augX suffix if it exists (e.g., 00001_aug1.jpg → 00001.jpg)
        match: Optional[re.Match] = re.match(r"^(.*)_aug\d+\.(\w+)$", filename)
        base_name: str = f"{match.group(1)}.{match.group(2)}" if match else filename

        # Look for original name in the map
        original: str = new_to_original.get(base_name, "desconocido")

        # Save the field in the sample
        sample["original_name"] = original
        sample.save()

        if (idx * 100 // total) != ((idx - 1) * 100 // total):
            printProgressBar(idx, total, prefix='Adding original names:', suffix='Complete', length=50)



def visualize_dataset(path: str, format: str, split: Optional[str], names: Optional[str] = None) -> None:
    """
    Visualizes a dataset in FiftyOne app.
    Args:
        format (str): Format of the dataset, either "yolo" or "coco".
        path (str): Path to the root folder of the dataset.
        split (str, optional): Split of the dataset (only for YOLO format).
        names (str, optional): Path to the original names CSV file (only for YOLO format).
    """

    dataset: fo.Dataset
    if format == "coco": 
        images_path: str = f"{path}/images"
        coco_path: str = f"{path}/labels_coco.json"
        dataset = load_coco_dataset(images_path, coco_path)

    elif format == "yolo": 
        yolo_path: str = f"{path}/yolo_dataset.yaml"
        dataset = load_yolo_dataset(path, yolo_path, split)

    else: 
        raise ValueError("Unsupported dataset format")

    # Add original names field if the dataset is in YOLO format 
    if names: 
        json_path: Path =  Path(__file__).parent.parent / "dataset" / "images" / "original_names_map.json"
        add_original_names_field(dataset, json_path)
    
    session = fo.launch_app(dataset)
    session.wait()



def arguments() -> Namespace:
    """Parse command line arguments for the script.
    Returns:
        Namespace: Parsed command line arguments.
    """ 
    parser=ArgumentParser(description="Visualize NN/CV project dataset")
    parser.add_argument("--path","--p",type=str,help="Path to the root folder of the yamls")
    parser.add_argument("--format","--f",type=str,choices=["yolo","coco"],help="Format of the dataset")
    parser.add_argument("--split","--s",required=False,type=str,choices=["train","val","test"],help="Split of the dataset (only for YOLO format)")
    parser.add_argument("--names","--n",required=False,type=str,help="Path to the original names JSON file (only for YOLO format)")
    args=parser.parse_args()
    return args



if __name__ == "__main__":
    args: Namespace = arguments()
    visualize_dataset(args.format, args.path, args.split, args.names)