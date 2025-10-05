# fiftyone_cli_visualizer.py
from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys

def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Visualize YOLO/COCO dataset with FiftyOne")
    parser.add_argument("--path", "--p", type=Path, required=True, help="Path to dataset root directory")
    parser.add_argument("--format", "--f", choices=["yolo", "coco"], required=True, help="Dataset format")
    parser.add_argument("--split", "--s", choices=["train", "val", "test"], required=False, default="val", help="YOLO split")
    parser.add_argument("--names", "--n", type=Path, required=False, help="Path to original names JSON")
    return parser.parse_args()


if __name__ == "__main__":
    # Ensure scripts directory is in sys.path so imports work
    scripts_dir = Path(__file__).parent.parent.resolve()
    sys.path.append(str(scripts_dir))

    from utils.config_logging import *
    setup_logging()
    
    from fiftyone_visualizer import FiftyOneVisualizer 

    try:
        args: Namespace = parse_arguments()

        visualizer = FiftyOneVisualizer(
            path=args.path,
            format=args.format,
            split=args.split,
            names_map_file=args.names
        )
        visualizer.visualize()

    except Exception as e:
        logger.exception(f"An error occurred during visualization: {e}")

