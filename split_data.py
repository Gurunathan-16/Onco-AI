import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, dest_base_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Split images from source_dir (with class subfolders) into train/val/test
    """
    random.seed(seed)
    
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    print(f"Found classes: {classes}")
    
    for cls in classes:
        class_path = os.path.join(source_dir, cls)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        
        total = len(images)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_images = images[:train_size]
        val_images = images[train_size:train_size + val_size]
        test_images = images[train_size + val_size:]
        
        # Create destination folders
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(dest_base_dir, split, cls)
            os.makedirs(split_dir, exist_ok=True)
        
        # Copy files
        for img in train_images:
            shutil.copy2(os.path.join(class_path, img), os.path.join(dest_base_dir, 'train', cls, img))
        for img in val_images:
            shutil.copy2(os.path.join(class_path, img), os.path.join(dest_base_dir, 'val', cls, img))
        for img in test_images:
            shutil.copy2(os.path.join(class_path, img), os.path.join(dest_base_dir, 'test', cls, img))
        
        print(f"Class '{cls}': {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

# ====================== USAGE ======================
if __name__ == "__main__":
    print("=== Data Splitting Tool for Breast & Oral Cancer ===\n")
    
    choice = input("Split for (1) Breast or (2) Oral? Enter 1 or 2: ").strip()
    
    if choice == "1":
        source = "dataset/breast"                    # Change if your raw folder path is different
        dest = "dataset_split/breast"
        print("Splitting Breast Ultrasound dataset...")
    elif choice == "2":
        source = "dataset/oral"
        dest = "dataset_split/oral"
        print("Splitting Oral Cancer dataset...")
    else:
        print("Invalid choice!")
        exit()
    
    split_dataset(source, dest, train_ratio=0.75, val_ratio=0.15)
    print("\n✅ Splitting completed! Check the 'processed' folder.")