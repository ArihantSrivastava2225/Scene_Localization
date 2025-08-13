import os
import zipfile
import requests
from pycocotools.coco import COCO

# URLs for MS COCO 2014 dataset
COCO_TRAIN_IMAGES = "http://images.cocodataset.org/zips/train2014.zip"
COCO_VAL_IMAGES = "http://images.cocodataset.org/zips/val2014.zip"
COCO_ANNOTATIONS = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

DATA_DIR = "data"
COCO_DIR = os.path.join(DATA_DIR, "coco")
REFCOCO_DIR = os.path.join(DATA_DIR, "refcoco")

def download_and_extract(url, dest_folder):
    filename = url.split("/")[-1]
    filepath = os.path.join(dest_folder, filename)

    if not os.path.exists(filepath):
        print(f"Downloading {filename} ...")
        r = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists, skipping download.")

    # Extract
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)
    print(f"Extracted {filename}")

def main():
    # Create main folders
    os.makedirs(COCO_DIR, exist_ok=True)
    os.makedirs(os.path.join(REFCOCO_DIR, "annotations"), exist_ok=True)

    # Download COCO train2014 images
    download_and_extract(COCO_TRAIN_IMAGES, COCO_DIR)

    # Download COCO val2014 images
    download_and_extract(COCO_VAL_IMAGES, COCO_DIR)

    # Download COCO annotations
    download_and_extract(COCO_ANNOTATIONS, COCO_DIR)

    print("\n✅ COCO data ready in:", COCO_DIR)
    print("📂 Drop your RefCOCO JSON files into:", os.path.join(REFCOCO_DIR, "annotations"))

if __name__ == "__main__":
    main()
