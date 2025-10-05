# file_system_manager.py
import json
from pathlib import Path
import shutil
from typing import Dict, List
from tqdm import tqdm
from utils.config_logging import logger


class FileSystemManager:
    """
    Manages basic filesystem operations like cleaning directories and 
    renaming/copying files for dataset preparation.
    """

    @staticmethod
    def clear_directory(path: Path, recursive: bool = False) -> None:
        """
        Deletes all files and optionally all subdirectories (recursively) 
        in the specified path.
        
        Args:
            path (Path): The directory to clear.
            recursive (bool): If True, deletes subdirectories and their contents using shutil.rmtree. 
                              If False, only deletes files and empty subdirectories.
        """
        if not path.exists() or not path.is_dir():
            logger.warning(f"Directory {path} does not exist or is not a directory. Skipping clear operation.")
            return

        files_iter: List[Path] = list(path.iterdir()) 
        total: int = len(files_iter)
        logger.info(f"Found {total} files/directories in {path} to process.")

        if total == 0:
            logger.info("Directory is already empty.")
            return

        if recursive:
            logger.warning(f"Deleting {path} recursively...")
            # Use shutil.rmtree for robust recursive deletion
            shutil.rmtree(path, ignore_errors=True)
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory {path} cleared recursively.")
            return

        # Non-recursive deletion (original behavior)
        for i, file in tqdm(
            enumerate(files_iter, start=1), 
            total=total, 
            desc='Deleting old files', 
            unit='item', 
            ncols=100
        ):
            try:
                if file.is_file() or file.is_symlink():
                    file.unlink() # Delete file
                elif file.is_dir():
                    # Attempt to remove directory only if empty (original logic)
                    try:
                        file.rmdir() 
                    except OSError:
                        pass # Directory is not empty, skip

            except Exception as e:
                logger.exception(f"Error deleting {file}: {e}")

        logger.info(f"Finished cleaning contents of directory: {path}")


    @staticmethod
    def rename_and_copy_images(
        input_dir: Path, 
        output_dir: Path, 
        output_json: Path
    ) -> Dict[str, str]:
        """
        Rename image files in the input directory to a consistent, sequential format 
        (e.g., '00001.jpg') and copy them to the output directory. 
        Saves the original_name -> new_name mapping to a JSON file.
        
        Args:
            input_dir (Path): The directory containing the original images.
            output_dir (Path): The directory where renamed images will be saved.
            output_json (Path): The path to save the JSON file mapping.

        Returns:
            Dict[str, str]: The mapping of original filename to new filename.
        """
        logger.info(f"Preparing to rename and copy images from {input_dir}...")
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        output_json.parent.mkdir(parents=True, exist_ok=True)

        # Filter and sort files for consistent ordering
        files: List[Path] = sorted([
            f for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])
        
        total: int = len(files)
        logger.info(f"Found {total} images to process.")

        if total == 0:
            return {}
            
        name_map: Dict[str, str] = {}
        
        for idx, file_path in tqdm(
            enumerate(files, start=1), 
            total=total, 
            desc='Renaming and Copying', 
            unit='file', 
            ncols=100
        ):
            ext: str = file_path.suffix.lower()
            # Consistent naming format (e.g., 00001.jpg)
            new_name: str = f"{idx:05d}{ext}" 
            new_path: Path = output_dir / new_name

            # Copy and rename simultaneously
            shutil.copy2(file_path, new_path) 
            name_map[file_path.name] = new_name

        # Save the mapping JSON file
        with open(output_json, 'w') as f:
            json.dump(name_map, f, indent=2)

        logger.info(f"Renaming complete. Mapping saved to {output_json}")
        return name_map