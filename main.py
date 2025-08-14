"""
Main entrypoint to train or infer.
Usage examples:
    python main.py --mode train --dataset_dir data/coco --year 2014 --split train
    python main.py --mode infer --image path/to/image.jpg --query "red bag" --ckpt checkpoints/model_epoch_5.h5
"""

import argparse
import numpy as np
import tensorflow as tf
import os

from train.train import train
from inference.infer import infer_and_visualize
from data.anchor_utils import generate_anchor_boxes
from models.grounding_model import VisualGroundingModel
#from data.image_utils import load_and_preprocess_image # Not used directly in main
#from data.text_utils import tokenize_and_pad # Not used directly in main
from transformers import BertTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'infer'], required=True)

    # Training arguments
    parser.add_argument('--dataset_dir', type=str, help='Path to COCO dataset root directory (contains train2014, val2014, annotations)')
    parser.add_argument('--year', type=str, default='2014', help='COCO dataset year (default: 2014)')
    parser.add_argument('--split', choices=['train', 'val'], default='train', help='Which split to use for training')

    # Inference arguments
    parser.add_argument('--image', type=str, help='Image path for inference')
    parser.add_argument('--query', type=str, help='Text query for inference')
    parser.add_argument('--ckpt', type=str, help='Checkpoint path to load model weights')

    args = parser.parse_args()

    # --- Constants for model architecture and anchors ---
    feat_h, feat_w = 7, 7
    image_size = (224, 224)
    # Note: anchor scales and aspect ratios should be consistent.
    # The training and inference code in your repo used different values.
    # We will use the same values for both now.
    anchor_scales = [0.5, 1.0, 2.0] # Relative scales
    anchor_ratios = [0.5, 1.0, 2.0] # Width/height ratios
    
    anchors_per_region = len(anchor_scales) * len(anchor_ratios)

    # FIX: Pre-compute anchors once for both train and infer modes
    anchors = generate_anchor_boxes(
        Hf=feat_h, 
        Wf=feat_w,
        anchors_per_region=anchors_per_region,
        image_height=image_size[0],
        image_width=image_size[1],
        scales=anchor_scales,
        aspect_ratios=anchor_ratios
    )
    # The anchors are in [y1, x1, y2, x2] format and normalized [0,1]

    kaggle_dataset_dir = '/kaggle/input/datasetscenelocalization/'
    train_year = '2014' # Hardcoding for train data.
    train_split = 'train' # Hardcoding for train data.

    if args.mode == 'train':
        annotations_path = os.path.join(kaggle_dataset_dir, 'annotations_trainval2014', 'annotations', f'instances_{train_split}{train_year}.json')
        images_dir = os.path.join(kaggle_dataset_dir, f'{train_split}{train_year}', f'{train_split}{train_year}')

        if not os.path.exists(annotations_path) or not os.path.exists(images_dir):
            raise FileNotFoundError(f"Kaggle data not found at {annotations_path} and {images_dir}")

        train(
            dataset_dir=kaggle_dataset_dir,
            year=train_year,
            split=train_split,
            epochs=50,
            batch_size=32,
            #save_dir='/kaggle/working/checkpointskaggle', # FIX: Use Kaggle's working directory
            save_dir='/data/coco',
            anchors=anchors
        )

    elif args.mode == 'infer':
        if args.image is None or args.query is None or args.ckpt is None:
            raise ValueError('Provide --image, --query, and --ckpt for inference')

        vocab_size = BertTokenizer.from_pretrained('bert-base-uncased').vocab_size
        max_query_len = 20
        num_regions = feat_h * feat_w
    
        model = VisualGroundingModel(
            vocab_size=vocab_size,
            num_regions=num_regions,
            anchors_per_region=anchors_per_region
        )
        model.build(input_shape=[(None, image_size[0], image_size[1], 3), (None, max_query_len), (None, max_query_len), (None, anchors.shape[0], 4)])
        model.load_weights(args.ckpt)

        decoded_boxes, scores = infer_and_visualize(
            model,
            args.image,
            args.query,
            image_size=image_size,
            feat_h=feat_h,
            feat_w=feat_w,
            anchors_per_region=anchors_per_region,
            anchors=anchors
        )


if __name__ == '__main__':
    main()
