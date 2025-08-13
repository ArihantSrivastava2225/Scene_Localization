"""
Training script for region-scoring + anchors visual grounding using RefCOCO.
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

# --- FIX: Re-correct the Dataset loader to avoid `[0]` index on tokenizer output ---
class RefCOCODataset():
    def __init__(self, ann_file, img_root, image_size=(224, 224), tokenizer=None):
        self.img_root = img_root
        self.image_size = image_size
        self.tokenizer = tokenizer

        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        self.annotations = coco_data['annotations']
        self.images = {img['id']: img for img in coco_data['images']}

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

            if 'sentences' in ann:
                query = ann['sentences'][0]['raw']
            elif 'caption' in ann:
                query = ann['caption']
            elif 'ref' in ann:
                query = ann['ref']
            elif image_id in self.captions_map:
                query = self.captions_map[image_id][0]
            else:
                query = "object"

            x, y, w, h = ann['bbox']
            x_min = x / image_info['width']
            y_min = y / image_info['height']
            x_max = (x + w) / image_info['width']
            y_max = (y + h) / image_info['height']
            bbox = [x_min, y_min, x_max, y_max]

            self.samples.append({
                'image_path': os.path.join(self.img_root, file_name),
                'query': query,
                'bbox': bbox
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['image_path']).convert("RGB").resize(self.image_size)
        img = tf.convert_to_tensor(np.array(img), dtype=tf.float32) / 255.0
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
            # FIX: Return the full tensors directly, without indexing.
            # The shape will be [1, 20] instead of [20].
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
        else:
            token_ids = [hash(w) % 10000 for w in query.lower().split()]
            input_ids = pad_sequences([token_ids], maxlen=20, padding='post', truncating='post')
            attention_mask = np.array([1 if t > 0 else 0 for t in input_ids[0]])
            attention_mask = np.expand_dims(attention_mask, axis=0)
        
        # We need to flatten the input_ids and attention_mask tensors
        input_ids = tf.squeeze(input_ids, axis=0)
        attention_mask = tf.squeeze(attention_mask, axis=0)

        return img, input_ids, attention_mask, bbox

# --- The collate_batch function is no longer needed with this dataset setup ---

def train(dataset_dir, year, split, epochs=50, batch_size=8, save_dir='checkpoints', anchors=None):
    if anchors is None:
        raise ValueError("The 'anchors' tensor must be passed to the train function.")

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    per_replica_batch_size = batch_size // strategy.num_replicas_in_sync

    image_size = (224, 224)
    feat_h, feat_w = 7, 7
    
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
    dataset_tf = dataset_tf.batch(per_replica_batch_size).prefetch(tf.data.AUTOTUNE)

    with strategy.scope():
        # Everything related to model creation and the training step must be in this scope
        vocab_size = tokenizer.vocab_size
        num_regions = feat_h * feat_w
        anchors_per_region = tf.shape(anchors)[0] // (feat_h * feat_w)
        
        model = VisualGroundingModel(vocab_size=vocab_size,
                                      num_regions=num_regions,
                                      anchors_per_region=anchors_per_region)
        optimizer = tf.keras.optimizers.Adam(1e-4)

        # FIX: The training step is now defined inside the strategy scope
        @tf.function
        def train_step(images, input_ids, attention_mask, gt_boxes):
            with tf.GradientTape() as tape:
                anchors_batched = tf.tile(tf.expand_dims(anchors, axis=0), [tf.shape(images)[0], 1, 1])
                gt_boxes_reshaped = tf.expand_dims(gt_boxes, axis=1)

                preds = model(images, input_ids, attention_mask, anchors_batched, training=True)
                scores = preds['scores']
                deltas = preds['deltas']
                
                total_loss, loss_info = matching_and_regression_loss(
                    scores, deltas, anchors_batched, gt_boxes_reshaped,
                    pos_iou_thresh=0.5, neg_iou_thresh=0.4, lambda_reg=5.0
                )

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            return total_loss, loss_info

    for epoch in range(epochs):
        print(f"\nStarting epoch {epoch + 1}/{epochs} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        avg_loss = tf.keras.metrics.Mean(name='total_loss')
        
        for step, (images, input_ids, attention_mask, gt_boxes) in enumerate(dataset_tf):
            total_loss, loss_info = train_step(images, input_ids, attention_mask, gt_boxes)
            
            avg_loss.update_state(total_loss)

            if step % 10 == 0:
                print(f"Step {step}: loss={total_loss:.4f}, bce={loss_info['bce']:.4f}, reg={loss_info['reg_loss']:.4f}")

        print(f"Epoch {epoch + 1} finished. Avg Loss: {avg_loss.result().numpy():.4f}")
        
        os.makedirs(save_dir, exist_ok=True)
        model.save_weights(os.path.join(save_dir, f'model_epoch_{epoch+1}.weights.h5'))
