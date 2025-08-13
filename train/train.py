"""
Training script for region-scoring + anchors visual grounding using RefCOCO.

Notes:
- This script expects a dataset converted to a CSV or a COCO/RefCOCO annotation file.
- For simplicity and robustness across environments, the data loader supports a small CSV format:
    image_path, x1, y1, x2, y2, query
  where x1,y1,x2,y2 are absolute pixel coordinates of the GT box.

- If you have the official RefCOCO JSON, convert it to the CSV of the above format
  (there are many small helper scripts online; conversion is one-time).

- The model used here imports VisualGroundingModel from models.grounding_model (assumed
  to exist as per the project scaffold we discussed).

"""

import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from pycocotools.coco import COCO
import json
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer


from models.grounding_model import VisualGroundingModel
from data.anchor_utils import generate_anchor_boxes, decode_boxes_from_deltas
from train.loss_functions import matching_and_regression_loss

class RefCOCODataset():
    def __init__(self, ann_file, img_root, image_size=(224, 224), tokenizer=None):
        self.img_root = img_root
        self.image_size = image_size
        self.tokenizer = tokenizer

        # Load main annotations (COCO-style instances file or RefCOCO)
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        self.annotations = coco_data['annotations']
        self.images = {img['id']: img for img in coco_data['images']}

        # Try to load matching captions file
        captions_file = ann_file.replace('instances_', 'captions_')
        if os.path.exists(captions_file):
            with open(captions_file, 'r') as f:
                captions_data = json.load(f)
            self.captions_map = {}
            for cap_ann in captions_data['annotations']:
                self.captions_map.setdefault(cap_ann['image_id'], []).append(cap_ann['caption'])
        else:
            self.captions_map = {}

        self.samples = []
        for ann in self.annotations:
            image_id = ann['image_id']
            image_info = self.images[image_id]
            file_name = image_info['file_name']

            # Get query text
            if 'sentences' in ann:  # RefCOCO style
                query = ann['sentences'][0]['raw']
            elif 'caption' in ann:  # Some custom formats
                query = ann['caption']
            elif 'ref' in ann:
                query = ann['ref']
            elif image_id in self.captions_map:  # COCO captions
                query = self.captions_map[image_id][0]
            else:
                query = "object"  # safe fallback
            # print(query)

            # COCO bbox is [x, y, width, height] in pixels
            x, y, w, h = ann['bbox']  
            
            # Convert to [x_min, y_min, x_max, y_max]
            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h
            
            image_width = image_info['width']
            image_height = image_info['height']
            # Normalize to [0, 1] relative to image size
            x_min /= image_width
            x_max /= image_width
            y_min /= image_height
            y_max /= image_height
            
            bbox = [x_min, y_min, x_max, y_max]  #converting to this format as iou would expect these instead of [x_center, y_center, width, height]

            self.samples.append({
                'image_path': os.path.join(self.img_root, file_name),
                'query': query,
                'bbox': bbox
            })
            # print(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['image_path']).convert("RGB").resize(self.image_size)
        img = tf.convert_to_tensor(np.array(img), dtype=tf.float32) / 255.0  # normalize to [0,1]
        query = sample['query']
        bbox = sample['bbox']
    
        if self.tokenizer:
            encoding = self.tokenizer(
                query,
                max_length=20,
                padding='max_length',
                truncation=True,
                return_tensors='tf',
            )
            input_ids = encoding['input_ids'][0]
            attention_mask = encoding['attention_mask'][0]
        else:
            token_ids = [hash(w) % 10000 for w in query.lower().split()]
            input_ids = pad_sequences([token_ids], maxlen=20, padding='post', truncating='post')[0]
            attention_mask = [1 if t > 0 else 0 for t in input_ids]
    
        return img, input_ids, attention_mask, bbox


def collate_batch(batch):
    images, input_ids_list, attention_masks_list, boxes = zip(*batch)
    images = tf.stack(images, axis=0)
    
    input_ids = np.stack(input_ids_list, axis=0)
    attention_mask = np.stack(attention_masks_list, axis=0)

    input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)
    attention_mask = tf.convert_to_tensor(attention_mask, dtype=tf.int32)
    boxes = tf.convert_to_tensor(np.stack(boxes, axis=0), dtype=tf.float32)

    return images, input_ids, attention_mask, boxes



def train(dataset_dir, year, split, epochs=10, batch_size=8, save_dir='checkpoints', anchors=None):
    """
    FIX: The function signature is updated to accept the pre-computed 'anchors' tensor.
    """
    # hyperparams
    image_size = (224, 224)
    feat_h, feat_w = 7, 7
    
    # FIX: We no longer generate anchors inside this function. We assume they are
    # passed from main.py and are a static tensor for the entire training run.
    if anchors is None:
        raise ValueError("The 'anchors' tensor must be passed to the train function.")

    # dataset setup (rest remains the same)
    ann_file = os.path.join(dataset_dir, 'annotations_trainval2014', 'annotations', f'instances_{split}{year}.json')
    img_root = os.path.join(dataset_dir, f'{split}{year}', f'{split}{year}')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = RefCOCODataset(ann_file, img_root, image_size=image_size, tokenizer=tokenizer)
    
    def gen():
        max_samples = 5000
        for i in range(min(len(dataset), max_samples)):
            yield dataset[i]
    
    out_types = (tf.float32, tf.int32, tf.int32, tf.float32)
    dataset_tf = tf.data.Dataset.from_generator(gen, output_types=out_types)
    dataset_tf = dataset_tf.batch(batch_size).map(lambda im, ids, mask, b: (im, ids, mask, b))
    dataset_tf = dataset_tf.prefetch(tf.data.AUTOTUNE)

    # model setup
    vocab_size = tokenizer.vocab_size
    num_regions = feat_h * feat_w
    # Assuming anchors_per_region is consistent
    anchors_per_region = tf.shape(anchors)[0] // (feat_h * feat_w)
    
    model = VisualGroundingModel(vocab_size=vocab_size,
                                  num_regions=num_regions,
                                  anchors_per_region=anchors_per_region)
    optimizer = tf.keras.optimizers.Adam(1e-4)

    for epoch in range(epochs):
        print(f"\nStarting epoch {epoch + 1}/{epochs} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        avg_loss = tf.keras.metrics.Mean(name='total_loss')
        
        for step, (images, input_ids, attention_mask, gt_boxes) in enumerate(dataset_tf):
            B = tf.shape(images)[0]
            
            # Tile the anchors for the current batch
            anchors_batched = tf.tile(tf.expand_dims(anchors, axis=0), [B, 1, 1])
            
            # Reshape gt_boxes to [B, G, 4] format (G=1 in this case)
            gt_boxes_reshaped = tf.expand_dims(gt_boxes, axis=1)

            with tf.GradientTape() as tape:
                preds = model(images, input_ids, attention_mask, anchors_batched, training=True)
                
                scores = preds['scores']
                deltas = preds['deltas']
                
                total_loss, loss_info = matching_and_regression_loss(
                    scores, deltas, anchors_batched, gt_boxes_reshaped,
                    pos_iou_thresh=0.5, neg_iou_thresh=0.4, lambda_reg=5.0
                )

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            avg_loss.update_state(total_loss)

            if step % 10 == 0:
                print(f"Step {step}: loss={total_loss:.4f}, bce={loss_info['bce']:.4f}, reg={loss_info['reg_loss']:.4f}")

        print(f"Epoch {epoch + 1} finished. Avg Loss: {avg_loss.result().numpy():.4f}")
        
        os.makedirs(save_dir, exist_ok=True)
        model.save_weights(os.path.join(save_dir, f'model_epoch_{epoch+1}.weights.h5'))