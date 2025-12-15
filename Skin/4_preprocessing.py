"""
Data Preprocessing and Augmentation Pipeline
This handles image loading, augmentation, and normalization
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import albumentations as A
from albumentations import Compose
import matplotlib.pyplot as plt

# Define augmentation pipeline
def get_training_augmentation():
    """
    Returns augmentation pipeline for training data
    Includes rotation, flips, brightness/contrast, blur
    """
    return Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Transpose(p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(blur_limit=(3, 7)),
        ], p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=0.3
        ),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

def get_validation_augmentation():
    """
    Returns augmentation for validation/test data
    Only resize and normalize, no random augmentations
    """
    return Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

def load_and_preprocess_image(image_path, augmentation=None):
    """
    Load image and apply augmentation
    
    Args:
        image_path: Path to image file
        augmentation: Albumentations Compose object
    
    Returns:
        Preprocessed image as numpy array
    """
    # Read image in RGB
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply augmentation
    if augmentation:
        augmented = augmentation(image=image)
        image = augmented['image']
    
    return image

def create_metadata_csv():
    """
    Create a CSV file with all image paths and labels
    Useful for tracking and debugging
    """
    data = []
    
    for split in ['train', 'val', 'test']:
        for label in ['benign', 'suspicious']:
            folder = Path(f'dataset/{split}/{label}')
            if folder.exists():
                for img_path in folder.glob('*.*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        data.append({
                            'filepath': str(img_path),
                            'split': split,
                            'label': label,
                            'label_binary': 0 if label == 'benign' else 1
                        })
    
    df = pd.DataFrame(data)
    df.to_csv('dataset/metadata.csv', index=False)
    
    print(f"✅ Created metadata.csv with {len(df)} images")
    print(f"   Train: {len(df[df['split']=='train'])}")
    print(f"   Val: {len(df[df['split']=='val'])}")
    print(f"   Test: {len(df[df['split']=='test'])}")
    print(f"   Benign: {len(df[df['label']=='benign'])}")
    print(f"   Suspicious: {len(df[df['label']=='suspicious'])}")
    
    return df

def visualize_augmentations(image_path, n_samples=5):
    """
    Visualize augmentation effects on a sample image
    """
    aug = get_training_augmentation()
    
    # Load original image
    original = cv2.imread(str(image_path))
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # Show original
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Show augmented versions
    for i in range(1, n_samples + 1):
        augmented = aug(image=original.copy())['image']
        # Denormalize for display
        img_display = augmented * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_display = np.clip(img_display, 0, 1)
        
        axes[i].imshow(img_display)
        axes[i].set_title(f'Augmented {i}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('reports/augmentation_examples.png', dpi=150, bbox_inches='tight')
    print("✅ Saved augmentation visualization to reports/augmentation_examples.png")
    plt.close()

def analyze_dataset_statistics(df):
    """
    Analyze and visualize dataset statistics
    """
    import seaborn as sns
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Class distribution per split
    split_counts = df.groupby(['split', 'label']).size().unstack()
    split_counts.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Class Distribution by Split')
    axes[0].set_xlabel('Split')
    axes[0].set_ylabel('Count')
    axes[0].legend(title='Label')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Overall class distribution
    label_counts = df['label'].value_counts()
    axes[1].pie(label_counts, labels=label_counts.index, autopct='%1.1f%%',
                colors=['#2ecc71', '#e74c3c'], startangle=90)
    axes[1].set_title('Overall Class Distribution')
    
    # Split distribution
    split_counts_total = df['split'].value_counts()
    axes[2].bar(split_counts_total.index, split_counts_total.values,
                color=['#3498db', '#9b59b6', '#f39c12'])
    axes[2].set_title('Images per Split')
    axes[2].set_xlabel('Split')
    axes[2].set_ylabel('Count')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/dataset_statistics.png', dpi=150, bbox_inches='tight')
    print("✅ Saved dataset statistics to reports/dataset_statistics.png")
    plt.close()

# Main execution
if __name__ == "__main__":
    import os
    os.makedirs('reports', exist_ok=True)
    
    print("🚀 Starting Data Preprocessing...")
    
    # Create metadata CSV
    df = create_metadata_csv()
    
    # Analyze statistics
    print("\n📊 Analyzing dataset statistics...")
    analyze_dataset_statistics(df)
    
    # Visualize augmentations on sample image
    print("\n🎨 Creating augmentation examples...")
    sample_images = list(Path('dataset/train/benign').glob('*.jpg'))
    if sample_images:
        visualize_augmentations(sample_images[0])
    
    print("\n✅ Preprocessing setup complete!")
    print("Next: Run 3_train_baseline.py to train the model")