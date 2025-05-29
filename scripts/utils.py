import json
from pathlib import Path
import shutil
from typing import Dict, List
from progress_bar import printProgressBar



def clear_directory(path: Path) -> None:
    """
    Delete all files and directories in the specified path.
    Args:
        path (Path): The directory to clear.
    """
    if not path.exists() or not path.is_dir():
        print(f"Directory {path} does not exist or is not a directory.")
        return
    
    files_iter: List[Path] = list(path.iterdir())  # Need list to measure and show progress
    total: int = len(files_iter)
    print(f"Found {total} files in {path}")

    if total == 0:
        print("No files to delete.")
        return
    
    printProgressBar(0, total, prefix='Deleting old files:', suffix='Complete', length=50)

    for i, file in enumerate(files_iter, start=1):
        try:
            if file.is_file() or file.is_symlink():
                file.unlink()

            elif file.is_dir():
                # Only remove empty directories to avoid recursive deletion errors
                try:
                    file.rmdir()
                except OSError:
                    pass  #< Not empty, skip or handle if recursive deletion is desired
        except Exception as e:
            print(f"Error deleting {file}: {e}")

        # Update progress bar only when visible percentage changes
        if (i * 100 // total) != ((i - 1) * 100 // total):
            printProgressBar(i, total, prefix='Deleting old files:', suffix='Complete', length=50)

    print(f"\nDeleted all the existing files in folder: {path}")



def rename_images(
    folder: Path, 
    renamed_folder: Path, 
    output_json_path: Path
) -> None:
    """
    Rename image files in the specified folder to a consistent format and save the mapping to a JSON file.
    Args:
        folder (Path): The folder containing the original images.
        renamed_folder (Path): The folder where renamed images will be saved.
        output_json_path (Path): The path to save the JSON file mapping original names to new names.
    """

    print(f"Sorting files in {folder}...")
    # consistent ordering
    files = sorted([
        f for f in folder.iterdir()
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ])
    print(f"Found {len(files)} images to rename.")

    name_map: Dict[str, str] = {}
    total: int = len(files)
    printProgressBar(0, total, prefix='Renaming files:', suffix='Complete', length=50)

    for idx, file_path in enumerate(files, start=1):
        ext: str = file_path.suffix.lower()
        new_name: str = f"{idx:05d}{ext}"
        new_path: Path= renamed_folder / new_name

        shutil.copy2(file_path, new_path) 
        name_map[file_path.name] = new_name
        # print(f"Rename: {file_path.name} → {new_name}")

        if (idx * 100 // total) != ((idx - 1) * 100 // total):
            printProgressBar(idx, total, prefix='Renaming files:', suffix='Complete', length=50)

    with open(output_json_path, 'w') as f:
        json.dump(name_map, f, indent=2)

    print(f"\nRenaming complete. Mapping saved to {output_json_path}")