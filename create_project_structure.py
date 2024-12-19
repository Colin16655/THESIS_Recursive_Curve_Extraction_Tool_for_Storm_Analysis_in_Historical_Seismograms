import os

def create_project_structure(base_path="seismogram_curve_extraction"):
    structure = {
        "data": ["raw", "processed", "ground_truth", "results"],
        "seismogram_extraction": {
            "": ["__init__.py", "data_generation.py", "preprocessing.py", "pipeline.py"],
            "filters": [
                "kalman_filter.py",
                "extended_kalman.py",
                "unscented_kalman.py",
                "particle_filter.py",
            ],
            "models": ["rnn.py", "lstm.py"],
            "utils": ["visualization.py", "metrics.py"],
        },
        "tests": [],
        "notebooks": [],
        "scripts": [],
        "docs": [],
        "config": [],
    }
    
    files_at_root = ["requirements.txt", "README.md"]

    def create_dir(base, sub_dirs):
        for sub_dir in sub_dirs:
            path = os.path.join(base, sub_dir)
            os.makedirs(path, exist_ok=True)

    def create_files(base, files):
        for file in files:
            path = os.path.join(base, file)
            with open(path, "w") as f:
                pass  # Create an empty file

    os.makedirs(base_path, exist_ok=True)
    create_files(base_path, files_at_root)

    for folder, sub_items in structure.items():
        if isinstance(sub_items, dict):  # Nested structure
            folder_path = os.path.join(base_path, folder)
            os.makedirs(folder_path, exist_ok=True)
            for sub_folder, files in sub_items.items():
                if sub_folder:  # Subfolder inside the nested folder
                    sub_folder_path = os.path.join(folder_path, sub_folder)
                    os.makedirs(sub_folder_path, exist_ok=True)
                    create_files(sub_folder_path, files)
                else:  # Files directly under the nested folder
                    create_files(folder_path, files)
        elif isinstance(sub_items, list):  # Flat folder with subfolders/files
            folder_path = os.path.join(base_path, folder)
            os.makedirs(folder_path, exist_ok=True)
            create_files(folder_path, sub_items)

if __name__ == "__main__":
    create_project_structure()
