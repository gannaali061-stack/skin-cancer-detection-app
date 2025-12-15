import os
import requests
from tqdm import tqdm
import zipfile

username = "janawaleed135"
key = "c4I5dac07d34b23b631d1I0e36a22e68"

print("Downloading dataset...")
os.makedirs('raw_data', exist_ok=True)

url = "https://www.kaggle.com/api/v1/datasets/download/nodoubttome/skin-cancer9-classesisic"

response = requests.get(url, auth=(username, key), stream=True)

if response.status_code == 200:
    total_size = int(response.headers.get('content-length', 0))
    zip_path = 'raw_data/dataset.zip'
    
    with open(zip_path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('raw_data')
    
    os.remove(zip_path)
    print("DONE!")
else:
    print(f"Error: Status code {response.status_code}")