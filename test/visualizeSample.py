import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # adds parent directory to path
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import BertTokenizer
from train.train import RefCOCODataset  # reuse your dataset class

def visualize_sample(img_tensor, query, bbox, image_size=(224, 224)):
    """
    Visualize one sample: image, bbox, and query.
    bbox is normalized [x_min, y_min, x_max, y_max]
    """
    img_np = np.array(img_tensor * 255, dtype=np.uint8)

    fig, ax = plt.subplots(1)
    ax.imshow(img_np)

    # Convert bbox to absolute pixel coords
    h, w = image_size
    x_min = bbox[0] * w
    y_min = bbox[1] * h
    x_max = bbox[2] * w
    y_max = bbox[3] * h

    rect = patches.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)
    ax.set_title(f"Query: {query}")
    plt.show()


if __name__ == "__main__":
    # CONFIG
    base_dir = r"C:\Users\hp\OneDrive\Desktop\ML\Projects\Scene_localization\Scene_Localization"
    dataset_dir = "data\\coco"   # root dataset folder
    year = "2014"
    split = "train"
    image_size = (224, 224)

    ann_file = os.path.join(base_dir, "data", "coco", "annotations", f"instances_{split}{year}.json")
    img_root = os.path.join(base_dir, "data", "coco", f"{split}{year}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    dataset = RefCOCODataset(
        ann_file=ann_file,
        img_root=img_root,
        image_size=image_size,
        tokenizer=tokenizer
    )

    print(f"Dataset size: {len(dataset)} samples\n")

    # Show first 5 samples
    for i in range(5):
        img, input_ids, attention_mask, bbox = dataset[i]
        query = dataset.samples[i]['query']
        img_path = dataset.samples[i]['image_path']

        print(f"Sample {i+1}")
        print(f"Image path: {img_path}")
        print(f"Query: {query}")
        print(f"Token IDs: {input_ids.numpy().tolist()}")
        print(f"Attention mask: {attention_mask.numpy().tolist()}")
        print(f"Normalized bbox: {bbox}")
        print("-" * 40)

        # Visualize
        visualize_sample(img, query, bbox, image_size=image_size)
