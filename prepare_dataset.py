import os
import shutil
from pathlib import Path

def prepare_dataset():
    input_folder = "brain_tumor_dataset"
    temp_folder = "temp_dataset"
    
    # First, create train and val directories
    train_dir = os.path.join(input_folder, "train")
    val_dir = os.path.join(input_folder, "val")
    
    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get list of all files in yes and no directories
    yes_files = list(Path(input_folder).glob("yes/*"))
    no_files = list(Path(input_folder).glob("no/*"))
    
    # Calculate split indices
    yes_split = int(len(yes_files) * 0.8)
    no_split = int(len(no_files) * 0.8)
    
    # Move files to train directory
    for f in yes_files[:yes_split]:
        dest_dir = os.path.join(train_dir, "yes")
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(f, dest_dir)
    
    for f in no_files[:no_split]:
        dest_dir = os.path.join(train_dir, "no")
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(f, dest_dir)
    
    # Move files to val directory
    for f in yes_files[yes_split:]:
        dest_dir = os.path.join(val_dir, "yes")
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(f, dest_dir)
    
    for f in no_files[no_split:]:
        dest_dir = os.path.join(val_dir, "no")
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(f, dest_dir)
    
    print("Dataset prepared successfully!")
    print(f"Training set: {yes_split} tumor images, {no_split} non-tumor images")
    print(f"Validation set: {len(yes_files)-yes_split} tumor images, {len(no_files)-no_split} non-tumor images")

if __name__ == "__main__":
    prepare_dataset() 