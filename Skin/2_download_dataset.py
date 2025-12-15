"""
تحميل ISIC Dataset من Kaggle
يجب تشغيل 1_setup_kaggle.py أولاً
"""
import os
import zipfile
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

def check_kaggle_setup():
    """التحقق من إعداد Kaggle API بشكل صحيح"""
    print("🔍 جاري التحقق من إعداد Kaggle API...")
    
    home_dir = os.path.expanduser("~")
    kaggle_json = os.path.join(home_dir, ".kaggle", "kaggle.json")
    
    if not os.path.exists(kaggle_json):
        print("❌ ملف kaggle.json غير موجود!")
        print("قم بتشغيل: python 1_setup_kaggle.py")
        return False
    
    print("✅ تم العثور على ملف kaggle.json")
    return True

def download_dataset():
    """تحميل Dataset من Kaggle"""
    
    if not check_kaggle_setup():
        return False
    
    print("\n" + "=" * 60)
    print("📥 تحميل ISIC DATASET")
    print("=" * 60)
    
    # إنشاء مجلد raw_data
    os.makedirs('raw_data', exist_ok=True)
    print("✅ تم إنشاء مجلد raw_data")
    
    # المصادقة على Kaggle API
    print("\n🔐 جاري المصادقة...")
    try:
        api = KaggleApi()
        api.authenticate()
        print("✅ تمت المصادقة بنجاح!")
    except Exception as e:
        print(f"❌ خطأ في المصادقة: {e}")
        return False
    
    # تحميل Dataset
    print("\n📦 جاري تحميل Dataset...")
    print("⏳ قد يستغرق هذا بعض الوقت (حجم كبير)...")
    
    try:
        dataset_name = "nodoubttome/skin-cancer9-classesisic"
        api.dataset_download_files(
            dataset_name,
            path='raw_data',
            unzip=True
        )
        print("✅ تم التحميل والاستخراج بنجاح!")
        
        # عرض محتويات المجلد
        print("\n📁 محتويات raw_data:")
        for item in os.listdir('raw_data'):
            item_path = os.path.join('raw_data', item)
            if os.path.isdir(item_path):
                print(f"  📂 {item}/")
            else:
                size = os.path.getsize(item_path) / (1024*1024)  # MB
                print(f"  📄 {item} ({size:.2f} MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ خطأ في التحميل: {e}")
        return False

def main():
    if download_dataset():
        print("\n" + "=" * 60)
        print("✅ اكتمل التحميل!")
        print("=" * 60)
        print("\n🎯 الخطوة التالية:")
        print("قم بتشغيل: python 3_organize_dataset.py")
    else:
        print("\n" + "=" * 60)
        print("❌ فشل التحميل")
        print("=" * 60)

if __name__ == "__main__":
    main()