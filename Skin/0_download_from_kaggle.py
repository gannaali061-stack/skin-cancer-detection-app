"""
Download ISIC dataset from Kaggle using API
Run this BEFORE 1_download_dataset.py
"""

import os
import zipfile
import shutil

def check_kaggle_setup():
    """Check if Kaggle API is properly configured"""
    print("🔍 Checking Kaggle API setup...")
    
    # Check if kaggle is installed
    try:
        import kaggle
        print("  ✅ Kaggle package installed")
    except ImportError:
        print("  ❌ Kaggle package not found!")
        print("  📥 Install it with: pip install kaggle")
        return False
    
    # Check if API key exists
    kaggle_json_path = os.path.expanduser('~/.kaggle/kaggle.json')
    if os.name == 'nt':  # Windows
        kaggle_json_path = os.path.join(os.environ['USERPROFILE'], '.kaggle', 'kaggle.json')
    
    if not os.path.exists(kaggle_json_path):
        print(f"  ❌ API key not found at: {kaggle_json_path}")
        print("\n📝 To fix this:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Scroll to 'API' section")
        print("  3. Click 'Create New Token'")
        print("  4. Place kaggle.json in the location above")
        return False
    else:
        print(f"  ✅ API key found at: {kaggle_json_path}")
    
    return True

def download_dataset():
    """Download the skin cancer dataset from Kaggle"""
    print("\n📥 Downloading dataset from Kaggle...")
    print("Dataset: nodoubtome/skin-cancer9-classesisic")
    print("Size: ~2 GB (this may take 5-15 minutes)")
    print("-" * 60)
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Initialize API
        api = KaggleApi()
        api.authenticate()
        
        print("✅ Authentication successful!")
        print("⏳ Starting download...")
        
        # Create raw_data folder if it doesn't exist
        os.makedirs('raw_data', exist_ok=True)
        
        # Download dataset
        api.dataset_download_files(
            'nodoubttome/skin-cancer9-classesisic',
            path='raw_data',
            unzip=True  # Automatically unzip
        )
        
        print("✅ Download complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {str(e)}")
        return False

def verify_download():
    """Verify that the dataset was downloaded correctly"""
    print("\n🔍 Verifying download...")
    
    # Look for the extracted folder
    expected_folders = [
        'raw_data/skin-cancer9-classesisic',
        'raw_data/actinic keratosis',  # Sometimes extracts directly
        'raw_data/melanoma'
    ]
    
    found = False
    dataset_path = None
    
    for folder in expected_folders:
        if os.path.exists(folder):
            found = True
            dataset_path = folder
            break
    
    if not found:
        # Check what's actually in raw_data
        print("📂 Contents of raw_data/:")
        if os.path.exists('raw_data'):
            contents = os.listdir('raw_data')
            for item in contents:
                print(f"  - {item}")
        
        print("\n⚠️ Dataset folder structure unexpected")
        return False
    
    print(f"✅ Dataset found at: {dataset_path}")
    
    # Count subfolders (disease classes)
    if os.path.isdir(dataset_path):
        subfolders = [f for f in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, f))]
        print(f"✅ Found {len(subfolders)} disease classes:")
        
        total_images = 0
        for folder in subfolders:
            folder_path = os.path.join(dataset_path, folder)
            images = [f for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            count = len(images)
            total_images += count
            print(f"  - {folder}: {count} images")
        
        print(f"\n📊 Total images: {total_images}")
        
        if total_images < 1000:
            print("⚠️ Warning: Less than expected number of images")
            return False
        
        return True
    
    return False

def organize_for_next_step():
    """
    Ensure the dataset is in the expected location for the next script
    """
    print("\n🔧 Organizing dataset structure...")
    
    # Check if dataset is directly in raw_data or in a subfolder
    if os.path.exists('raw_data/actinic keratosis'):
        # Dataset extracted directly to raw_data
        print("  ℹ️ Dataset is already in raw_data/")
        print("  ✅ Ready for next step!")
        return True
    
    elif os.path.exists('raw_data/skin-cancer9-classesisic'):
        # Dataset is in subfolder
        print("  ℹ️ Dataset is in subfolder: skin-cancer9-classesisic")
        print("  ✅ Ready for next step!")
        return True
    
    else:
        # Look for zip file
        zip_files = [f for f in os.listdir('raw_data') if f.endswith('.zip')]
        if zip_files:
            print(f"  ⚠️ Found zip file: {zip_files[0]}")
            print("  📦 Extracting...")
            
            zip_path = os.path.join('raw_data', zip_files[0])
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('raw_data')
            
            print("  ✅ Extraction complete!")
            
            # Remove zip file
            os.remove(zip_path)
            print("  🗑️  Removed zip file")
            
            return True
        
        print("  ❌ Could not find dataset")
        return False

def main():
    """Main execution function"""
    print("=" * 60)
    print("🚀 KAGGLE DATASET DOWNLOADER")
    print("=" * 60)
    
    # Step 1: Check setup
    if not check_kaggle_setup():
        print("\n" + "=" * 60)
        print("❌ SETUP INCOMPLETE")
        print("=" * 60)
        print("\nPlease complete the setup steps above and run again.")
        return
    
    # Step 2: Download
    if not download_dataset():
        print("\n" + "=" * 60)
        print("❌ DOWNLOAD FAILED")
        print("=" * 60)
        return
    
    # Step 3: Verify
    if not verify_download():
        print("\n" + "=" * 60)
        print("⚠️  VERIFICATION ISSUES")
        print("=" * 60)
        print("\nDataset may have downloaded but structure is unexpected.")
        print("Check the raw_data/ folder manually.")
        return
    
    # Step 4: Organize
    if not organize_for_next_step():
        print("\n" + "=" * 60)
        print("⚠️  ORGANIZATION ISSUES")
        print("=" * 60)
        return
    
    # Success!
    print("\n" + "=" * 60)
    print("✅ SUCCESS! DATASET READY")
    print("=" * 60)
    print("\n🎯 Next steps:")
    print("  1. Run: python 1_download_dataset_KAGGLE.py")
    print("  2. Run: python 2_preprocessing.py")
    print("  3. Run: python 3_train_baseline.py")
    print("  4. Run: python 4_evaluate_model.py")
    print("\n🎉 Happy training!")

if __name__ == "__main__":
    main()