import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO

# Paths (adjust to your project root if needed)
DATA_DIR = "data\\coco"
base_dir = r"C:\Users\hp\OneDrive\Desktop\ML\Projects\Scene_localization\Scene_Localization"
ANN_FILE = os.path.join(base_dir, "data", "coco", "annotations", "instances_train2014.json")
IMG_DIR = os.path.join(base_dir, "data", "coco", "train2014")

# Load COCO dataset
print(f"Loading annotations from {ANN_FILE}")
coco = COCO(ANN_FILE)

# Get all image IDs
img_ids = coco.getImgIds()

# Pick a random image
image_id = random.choice(img_ids)
img_info = coco.loadImgs(image_id)[0]
img_path = os.path.join(IMG_DIR, img_info["file_name"])

# Get annotation IDs for that image
ann_ids = coco.getAnnIds(imgIds=image_id)
anns = coco.loadAnns(ann_ids)

print("\n--- SAMPLE ---")
print("Image file:", img_info["file_name"])
print("Image size:", img_info["width"], "x", img_info["height"])
print(f"Number of annotations: {len(anns)}")

# Load image
img = plt.imread(img_path)
fig, ax = plt.subplots(1)
ax.imshow(img)

# Draw bounding boxes with labels
for ann in anns:
    x, y, w, h = ann["bbox"]
    cat_id = ann["category_id"]
    cat_name = coco.loadCats(cat_id)[0]["name"]
    rect = patches.Rectangle(
        (x, y), w, h,
        linewidth=2, edgecolor='r', facecolor='none'
    )
    ax.add_patch(rect)
    ax.text(x, y, cat_name, color='white', fontsize=8,
            bbox=dict(facecolor='red', alpha=0.5, pad=1))

plt.axis('off')
plt.show()
