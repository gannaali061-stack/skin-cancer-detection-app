"""
Kaggle API Setup Script
هذا السكريبت بيعمل إعداد أولي لـ Kaggle API
"""
import os
import json

def setup_kaggle_credentials():
    """إنشاء ملف kaggle.json مع بيانات الاعتماد"""
    
    credentials = {
        "username": "janawaleed135",
        "key": "c4I5dac07d34b23b631d1I0e36a22e68"
    }
    
    # الحصول على مسار المجلد الرئيسي
    home_dir = os.path.expanduser("~")
    kaggle_dir = os.path.join(home_dir, ".kaggle")
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    
    print("🔧 جاري إعداد Kaggle API...")
    print(f"المجلد الرئيسي: {home_dir}")
    print(f"مجلد Kaggle: {kaggle_dir}")
    
    # إنشاء مجلد .kaggle إذا لم يكن موجوداً
    try:
        os.makedirs(kaggle_dir, exist_ok=True)
        print(f"✅ تم إنشاء المجلد: {kaggle_dir}")
    except Exception as e:
        print(f"❌ خطأ في إنشاء المجلد: {e}")
        return False
    
    # كتابة ملف kaggle.json
    try:
        with open(kaggle_json_path, 'w') as f:
            json.dump(credentials, f, indent=2)
        print(f"✅ تم إنشاء kaggle.json في: {kaggle_json_path}")
        
        # تعيين الصلاحيات (مهم للأمان)
        if os.name != 'nt':  # Unix/Linux/Mac
            os.chmod(kaggle_json_path, 0o600)
            print("✅ تم تعيين الصلاحيات بشكل آمن")
            
    except Exception as e:
        print(f"❌ خطأ في إنشاء kaggle.json: {e}")
        return False
    
    # التحقق من إنشاء الملف
    if os.path.exists(kaggle_json_path):
        print("✅ تم التحقق بنجاح!")
        print(f"✅ حجم الملف: {os.path.getsize(kaggle_json_path)} بايت")
        with open(kaggle_json_path, 'r') as f:
            content = json.load(f)
        print(f"✅ اسم المستخدم: {content['username']}")
        print(f"✅ المفتاح: {'*' * 20} (مخفي)")
        return True
    else:
        print("❌ لم يتم إنشاء الملف!")
        return False

def test_kaggle_api():
    """اختبار عمل Kaggle API"""
    print("\n🧪 جاري اختبار Kaggle API...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("✅ تم المصادقة على Kaggle API بنجاح!")
        return True
    except ImportError:
        print("⚠️ حزمة Kaggle غير مثبتة")
        print("قم بتشغيل: pip install kaggle")
        return False
    except Exception as e:
        print(f"❌ فشلت المصادقة: {e}")
        return False

def main():
    print("=" * 60)
    print("🚀 إعداد KAGGLE API")
    print("=" * 60)
    
    if setup_kaggle_credentials():
        print("\n" + "=" * 60)
        print("✅ اكتمل الإعداد!")
        print("=" * 60)
        test_kaggle_api()
        
        print("\n🎯 الخطوات التالية:")
        print("1. إذا رأيت 'حزمة Kaggle غير مثبتة'، قم بتشغيل:")
        print("   pip install kaggle")
        print("2. ثم قم بتشغيل:")
        print("   python 2_download_dataset.py")
    else:
        print("\n" + "=" * 60)
        print("❌ فشل الإعداد")
        print("=" * 60)
        print("\nحاول التشغيل كمسؤول أو تحقق من الصلاحيات.")

if __name__ == "__main__":
    main()