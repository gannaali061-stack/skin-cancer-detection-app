"""
1_organize_dataset.py
Organize ISIC dataset from 9 classes to binary classification
"""
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def create_directory_structure():
    """Create directory structure for organized dataset"""
    print("Creating directory structure...")
    
    dirs = [
        'dataset/train/benign',
        'dataset/train/malignant',
        'dataset/val/benign',
        'dataset/val/malignant',
        'dataset/test/benign',
        'dataset/test/malignant'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("Directory structure created successfully")

def get_class_mapping():
    """Define malignant and benign classes"""
    
    malignant = {
        'melanoma': 'malignant',
        'basal cell carcinoma': 'malignant',
        'squamous cell carcinoma': 'malignant'
    }
    
    benign = {
        'actinic keratosis': 'benign',
        'dermatofibroma': 'benign',
        'nevus': 'benign',
        'pigmented benign keratosis': 'benign',
        'seborrheic keratosis': 'benign',
        'vascular lesion': 'benign'
    }
    
    return {**malignant, **benign}

def process_folder(source_path, target_split, class_mapping, stats, train_ratio=0.8):
    """Process images from source folder"""
    
    if not source_path.exists():
        print(f"Warning: {source_path} does not exist")
        return
    
    for class_folder in source_path.iterdir():
        if not class_folder.is_dir():
            continue
        
        folder_name = class_folder.name.lower()
        
        if folder_name not in class_mapping:
            print(f"Skipping unknown class: {folder_name}")
            continue
        
        target_class = class_mapping[folder_name]
        
        images = list(class_folder.glob('*.jpg')) + \
                 list(class_folder.glob('*.png')) + \
                 list(class_folder.glob('*.jpeg'))
        
        if not images:
            print(f"No images found in: {folder_name}")
            continue
        
        if target_split == 'train':
            random.shuffle(images)
            split_idx = int(len(images) * train_ratio)
            train_images = images[:split_idx]
            val_images = images[split_idx:]
            
            for img in tqdm(train_images, desc=f"Train/{folder_name}"):
                dest = f"dataset/train/{target_class}/{img.name}"
                shutil.copy2(img, dest)
                stats['train'][target_class] += 1
            
            for img in tqdm(val_images, desc=f"Val/{folder_name}"):
                dest = f"dataset/val/{target_class}/{img.name}"
                shutil.copy2(img, dest)
                stats['val'][target_class] += 1
        
        else:
            for img in tqdm(images, desc=f"Test/{folder_name}"):
                dest = f"dataset/test/{target_class}/{img.name}"
                shutil.copy2(img, dest)
                stats['test'][target_class] += 1

def print_statistics(stats):
    """Print dataset statistics"""
    
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    print("\nTRAIN:")
    print(f"  Benign: {stats['train']['benign']}")
    print(f"  Malignant: {stats['train']['malignant']}")
    print(f"  Total: {stats['train']['benign'] + stats['train']['malignant']}")
    
    print("\nVALIDATION:")
    print(f"  Benign: {stats['val']['benign']}")
    print(f"  Malignant: {stats['val']['malignant']}")
    print(f"  Total: {stats['val']['benign'] + stats['val']['malignant']}")
    
    print("\nTEST:")
    print(f"  Benign: {stats['test']['benign']}")
    print(f"  Malignant: {stats['test']['malignant']}")
    print(f"  Total: {stats['test']['benign'] + stats['test']['malignant']}")
    
    total_benign = sum(s['benign'] for s in stats.values())
    total_malignant = sum(s['malignant'] for s in stats.values())
    
    print("\n" + "="*60)
    print("TOTAL:")
    print(f"  Benign: {total_benign}")
    print(f"  Malignant: {total_malignant}")
    print(f"  Grand Total: {total_benign + total_malignant}")
    print("="*60)

def main():
    print("="*60)
    print("ORGANIZING ISIC DATASET")
    print("="*60)
    
    source_folder = Path('raw_data/Skin cancer ISIC The International Skin Imaging Collaboration')
    
    if not source_folder.exists():
        print(f"ERROR: Source folder not found: {source_folder}")
        print("Please check the path in raw_data folder")
        return
    
    create_directory_structure()
    
    random.seed(42)
    
    stats = {
        'train': {'benign': 0, 'malignant': 0},
        'val': {'benign': 0, 'malignant': 0},
        'test': {'benign': 0, 'malignant': 0}
    }
    
    class_mapping = get_class_mapping()
    
    print("\nProcessing Train folder (will split into train/val)...")
    train_path = source_folder / 'Train'
    process_folder(train_path, 'train', class_mapping, stats, train_ratio=0.8)
    
    print("\nProcessing Test folder...")
    test_path = source_folder / 'Test'
    process_folder(test_path, 'test', class_mapping, stats)
    
    print_statistics(stats)
    
    print("\nDataset organization completed successfully!")
    print("\nNext step: python 2_preprocess_data.py")

if __name__ == "__main__":
    main()