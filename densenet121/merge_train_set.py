import os
import shutil

# Paths to the four parent folders
parent_folders = {
    "deck1": "./jackgpt_dataset/processed/deck1",
    "deck2": "./jackgpt_dataset/processed/deck2",
    "deck3": "./jackgpt_dataset/processed/deck3",
    "deck4": "./jackgpt_dataset/processed/deck4"
}

# Path to the new combined folder
combined_folder = "./dataset/train"

# Ensure the combined folder exists
os.makedirs(combined_folder, exist_ok=True)

# Get the list of subfolder names (assume all parent folders have the same structure)
subfolder_names = os.listdir(next(iter(parent_folders.values())))

# Iterate over each subfolder name
for subfolder in subfolder_names:
    subfolder_path = os.path.join(combined_folder, subfolder)
    os.makedirs(subfolder_path, exist_ok=True)  # Create subfolder in combined folder

    for parent_name, parent_path in parent_folders.items():
        source_subfolder = os.path.join(parent_path, subfolder)
        if os.path.exists(source_subfolder):
            for item in os.listdir(source_subfolder):
                src = os.path.join(source_subfolder, item)
                # Generate a unique file name: parentName_originalFileName
                unique_name = f"{parent_name}_{item}"
                dst = os.path.join(subfolder_path, unique_name)

                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)