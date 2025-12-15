"""
2_preprocess_data.py
Data preprocessing and quality checks
"""
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self, img_size=224):
        self.img_size = img_size
    
    def create_dataset_info(self, dataset_dir='dataset'):
        """Create CSV file with dataset information"""
        
        print("Creating dataset info file...")
        
        data = []
        
        for split in ['train', 'val', 'test']:
            split_path = Path(dataset_dir) / split
            
            for class_name in ['benign', 'malignant']:
                class_path = split_path / class_name
                
                if not class_path.exists():
                    continue
                
                images = list(class_path.glob('*.jpg')) + \
                         list(class_path.glob('*.png')) + \
                         list(class_path.glob('*.jpeg'))
                
                for img_path in images:
                    data.append({
                        'image_path': str(img_path),
                        'split': split,
                        'class': class_name,
                        'label': 1 if class_name == 'malignant' else 0
                    })
        
        df = pd.DataFrame(data)
        df.to_csv('dataset_info.csv', index=False)
        
        print(f"Created dataset_info.csv with {len(df)} images")
        
        print("\nDataset Statistics:")
        print(df.groupby(['split', 'class']).size())
        
        return df
    
    def visualize_samples(self, df, n_samples=6):
        """Visualize random samples from dataset"""
        
        print(f"\nVisualizing {n_samples} random samples...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        samples = df.sample(n=n_samples)
        
        for idx, (_, row) in enumerate(samples.iterrows()):
            img = cv2.imread(row['image_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[idx].imshow(img)
            axes[idx].set_title(
                f"{row['class'].upper()}\n{row['split']}",
                fontsize=12,
                fontweight='bold',
                color='red' if row['class'] == 'malignant' else 'green'
            )
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
        print("Saved sample_images.png")
        plt.close()
    
    def check_image_quality(self, dataset_dir='dataset'):
        """Check image quality and sizes"""
        
        print("\nChecking image quality...")
        
        issues = []
        sizes = []
        
        for split in ['train', 'val', 'test']:
            split_path = Path(dataset_dir) / split
            
            for class_name in ['benign', 'malignant']:
                class_path = split_path / class_name
                
                if not class_path.exists():
                    continue
                
                images = list(class_path.glob('*.jpg')) + \
                         list(class_path.glob('*.png'))
                
                for img_path in tqdm(images, desc=f"Checking {split}/{class_name}"):
                    img = cv2.imread(str(img_path))
                    
                    if img is None:
                        issues.append(f"Cannot read: {img_path}")
                        continue
                    
                    h, w = img.shape[:2]
                    sizes.append((w, h))
                    
                    if w < 50 or h < 50:
                        issues.append(f"Too small ({w}x{h}): {img_path}")
        
        print(f"\nChecked {len(sizes)} images")
        
        if issues:
            print(f"\nFound {len(issues)} issues:")
            for issue in issues[:10]:
                print(f"  - {issue}")
        else:
            print("All images are valid!")
        
        if sizes:
            widths, heights = zip(*sizes)
            print(f"\nImage Size Statistics:")
            print(f"  Width: min={min(widths)}, max={max(widths)}, avg={np.mean(widths):.1f}")
            print(f"  Height: min={min(heights)}, max={max(heights)}, avg={np.mean(heights):.1f}")

def main():
    print("="*60)
    print("DATA PREPROCESSING")
    print("="*60)
    
    if not os.path.exists('dataset'):
        print("ERROR: dataset folder not found!")
        print("Run: python 1_organize_dataset.py first")
        return
    
    preprocessor = DataPreprocessor(img_size=224)
    
    df = preprocessor.create_dataset_info()
    
    preprocessor.visualize_samples(df)
    
    preprocessor.check_image_quality()
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETED")
    print("="*60)
    print("\nNext step: python 3_train_model.py")

if __name__ == "__main__":
    main()