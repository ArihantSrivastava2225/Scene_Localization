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
from train.loss_functions import matching_and_regression_loss, iou_boxes, iou_box

# --- The RefCOCODataset class has a minor fix for consistency ---
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
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
        else:
            token_ids = [hash(w) % 10000 for w in query.lower().split()]
            input_ids = pad_sequences([token_ids], maxlen=20, padding='post', truncating='post')
            attention_mask = np.array([1 if t > 0 else 0 for t in input_ids[0]])
            attention_mask = np.expand_dims(attention_mask, axis=0)
        
        input_ids = tf.squeeze(input_ids, axis=0)
        attention_mask = tf.squeeze(attention_mask, axis=0)

        return img, input_ids, attention_mask, bbox

# --- The collate_batch function is no longer needed with this dataset setup ---

def train(dataset_dir, year, split, epochs=10, batch_size=8, save_dir='checkpoints', anchors=None):
    if anchors is None:
        raise ValueError("The 'anchors' tensor must be passed to the train function.")

    image_size = (224, 224)
    feat_h, feat_w = 7, 7
    
    ann_file = os.path.join(dataset_dir, 'annotations_trainval2014', 'annotations', f'instances_{split}{year}.json')
    img_root = os.path.join(dataset_dir, f'{split}{year}', f'{split}{year}')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = RefCOCODataset(ann_file, img_root, image_size=image_size, tokenizer=tokenizer)
    
    def gen():
        max_samples = 10000
        for i in range(min(len(dataset), max_samples)):
            yield dataset[i]

    # FIX: Calculate steps per epoch manually
    max_samples_in_training = 10000
    steps_per_epoch = max_samples_in_training // batch_size

    out_types = (tf.float32, tf.int32, tf.int32, tf.float32)
    output_shapes = (tf.TensorShape(image_size + (3,)), tf.TensorShape([20]), tf.TensorShape([20]), tf.TensorShape([4]))
    
    dataset_tf = tf.data.Dataset.from_generator(gen, output_types=out_types, output_shapes=output_shapes)
    dataset_tf = dataset_tf.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    vocab_size = tokenizer.vocab_size
    num_regions = feat_h * feat_w
    anchors_per_region = tf.shape(anchors)[0] // (feat_h * feat_w)
    
    model = VisualGroundingModel(vocab_size=vocab_size,
                                  num_regions=num_regions,
                                  anchors_per_region=anchors_per_region)
    
    # Build the model with a dummy input to initialize weights
    dummy_image = tf.random.uniform((1, image_size[0], image_size[1], 3))
    dummy_ids = tf.zeros((1, 20), dtype=tf.int32)
    dummy_mask = tf.zeros((1, 20), dtype=tf.int32)
    dummy_anchors = tf.zeros((1, anchors.shape[0], 4))
    _ = model(dummy_image, dummy_ids, dummy_mask, dummy_anchors)

    # --- Training step function ---
    @tf.function
    def train_step(images, input_ids, attention_mask, gt_boxes, optimizer):
        with tf.GradientTape() as tape:
            anchors_batched = tf.tile(tf.expand_dims(anchors, axis=0), [tf.shape(images)[0], 1, 1])
            gt_boxes_reshaped = tf.expand_dims(gt_boxes, axis=1)

            preds = model(images, input_ids, attention_mask, anchors_batched, training=True)
            scores = preds['scores']
            deltas = preds['deltas']
            raw_scores = preds['raw_scores']
            
            total_loss, loss_info = matching_and_regression_loss(
                scores, deltas, anchors_batched, gt_boxes_reshaped, raw_scores, 
                pos_iou_thresh=0.5, neg_iou_thresh=0.4, lambda_reg=5.0
            )

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        return total_loss, loss_info

    # --- Phase 1: Train with frozen encoders (for a few epochs) ---
    initial_epochs = 10
    initial_learning_rate = 1e-4
    print(f"\n--- Starting Phase 1: Training top layers with frozen encoders (for {initial_epochs} epochs) ---")
    
    # Ensure encoders are frozen
    model.image_encoder.trainable = False
    model.text_encoder.trainable = False

    print(f"Number of trainable variables: {len(model.trainable_variables)}")
    # FIX: Use a learning rate schedule for the first phase
    # --- Combined learning rate schedule for both phases ---
    # FIX: Use a single, combined schedule
    boundaries = [steps_per_epoch * 10]
    values = [1e-4, 1e-5]
    learning_rate_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    
    # FIX: Create the optimizer once with the schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

    for epoch in range(initial_epochs):
        print(f"\nStarting epoch {epoch + 1}/{initial_epochs} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        avg_loss = tf.keras.metrics.Mean(name='total_loss')
        
        for step, (images, input_ids, attention_mask, gt_boxes) in enumerate(dataset_tf):
            total_loss, loss_info = train_step(images, input_ids, attention_mask, gt_boxes, optimizer)
            avg_loss.update_state(total_loss)

            if step % 10 == 0:
                print(f"Step {step}: loss={total_loss:.4f}, focal={loss_info['focal']:.4f}, con={loss_info['contrastive']:.4f}")
        
        print(f"Epoch {epoch + 1} finished. Avg Loss: {avg_loss.result().numpy():.4f}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_weights(os.path.join(save_dir, f'model_epoch_{epoch+1}.weights.h5'))
            # --- Evaluation on Validation Set ---
        print("\n--- Evaluating model on validation set ---")
        val_ann_file = os.path.join(dataset_dir, 'annotations_trainval2014', 'annotations', f'instances_val{year}.json')
        val_img_root = os.path.join(dataset_dir, f'val{year}', f'val{year}')
        
        val_dataset = RefCOCODataset(val_ann_file, val_img_root, image_size=image_size, tokenizer=tokenizer)
        
        def val_gen():
            # Use a reasonable number of samples for validation, e.g., all of them or a subset
            max_val_samples = min(len(val_dataset), 1000) # Limit validation to 1000 samples for speed
            for i in range(max_val_samples):
                yield val_dataset[i]
    
        val_dataset_tf = tf.data.Dataset.from_generator(
            val_gen, output_types=out_types, output_shapes=output_shapes
        )
        val_dataset_tf = val_dataset_tf.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
        total_iou = 0.0
        num_predictions = 0
    
        for step, (images, input_ids, attention_mask, gt_boxes) in enumerate(val_dataset_tf):
            B = tf.shape(images)[0]
            anchors_batched = tf.tile(tf.expand_dims(anchors, axis=0), [B, 1, 1])
            
            # Run inference
            preds = model(images, input_ids, attention_mask, anchors_batched, training=False)
            raw_scores = preds['scores']
            deltas = preds['deltas']
    
            # Convert scores to probabilities
            scores_prob = tf.sigmoid(raw_scores)
    
            # FIX: Loop through each image in the batch to decode and evaluate
            for i in range(B):
                # Decode boxes for the current image `i`
                # We need to pass the specific anchors and deltas for this image
                # `anchors_batched[i]` and `deltas[i]` are the correct tensors to use
                decoded_boxes_norm_i = tf.numpy_function(
                    decode_boxes_from_deltas,
                    [anchors_batched[i], deltas[i]],
                    Tout=tf.float32
                )
                decoded_boxes_norm_i.set_shape((anchors_batched.shape[1], 4))
    
                # Select the top-scoring box for the current image
                best_anchor_idx = tf.argmax(scores_prob[i])
                predicted_box = decoded_boxes_norm_i[best_anchor_idx]
                
                gt_box = gt_boxes[i]
    
                iou_val = iou_box(tf.expand_dims(predicted_box, axis=0), tf.expand_dims(gt_box, axis=0))
                total_iou += iou_val[0,0]
                num_predictions += 1
                
                if step % 10 == 0 and i == 0:
                    print(f"Validation Step {step}, Image {i}: IoU = {iou_val[0,0]:.4f}")
    
        if num_predictions > 0:
            average_iou = total_iou / num_predictions
            print(f"\nAverage IoU on validation set: {average_iou:.4f}")
        else:
            print("\nNo predictions made on validation set.")


    # --- Phase 2: Unfreeze and fine-tune encoders (for remaining epochs) ---
    fine_tuning_epochs = epochs - initial_epochs
    fine_tuning_learning_rate = 1e-5
    print(f"\n--- Starting Phase 2: Fine-tuning encoders with low learning rate (for {fine_tuning_epochs} epochs) ---")
    
    # Unfreeze encoders for fine-tuning
    model.image_encoder.trainable = True
    model.text_encoder.trainable = True

    print(f"Number of trainable variables: {len(model.trainable_variables)}")
    
    for epoch in range(initial_epochs, epochs):
        print(f"\nStarting epoch {epoch + 1}/{epochs} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        avg_loss = tf.keras.metrics.Mean(name='total_loss')
        
        for step, (images, input_ids, attention_mask, gt_boxes) in enumerate(dataset_tf):
            total_loss, loss_info = train_step(images, input_ids, attention_mask, gt_boxes, optimizer)
            avg_loss.update_state(total_loss)

            if step % 10 == 0:
                print(f"Step {step}: loss={total_loss:.4f}, focal={loss_info['focal']:.4f}, con={loss_info['contrastive']:.4f}")
        
        print(f"Epoch {epoch + 1} finished. Avg Loss: {avg_loss.result().numpy():.4f}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_weights(os.path.join(save_dir, f'model_epoch_{epoch+1}.weights.h5'))
        # --- Evaluation on Validation Set ---
        print("\n--- Evaluating model on validation set ---")
        val_ann_file = os.path.join(dataset_dir, 'annotations_trainval2014', 'annotations', f'instances_val{year}.json')
        val_img_root = os.path.join(dataset_dir, f'val{year}', f'val{year}')
        
        val_dataset = RefCOCODataset(val_ann_file, val_img_root, image_size=image_size, tokenizer=tokenizer)
        
        def val_gen():
            # Use a reasonable number of samples for validation, e.g., all of them or a subset
            max_val_samples = min(len(val_dataset), 1000) # Limit validation to 1000 samples for speed
            for i in range(max_val_samples):
                yield val_dataset[i]
    
        val_dataset_tf = tf.data.Dataset.from_generator(
            val_gen, output_types=out_types, output_shapes=output_shapes
        )
        val_dataset_tf = val_dataset_tf.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
        total_iou = 0.0
        num_predictions = 0
    
        for step, (images, input_ids, attention_mask, gt_boxes) in enumerate(val_dataset_tf):
            B = tf.shape(images)[0]
            anchors_batched = tf.tile(tf.expand_dims(anchors, axis=0), [B, 1, 1])
            
            # Run inference
            preds = model(images, input_ids, attention_mask, anchors_batched, training=False)
            raw_scores = preds['scores']
            deltas = preds['deltas']
    
            # Convert scores to probabilities
            scores_prob = tf.sigmoid(raw_scores)
    
            # FIX: Loop through each image in the batch to decode and evaluate
            for i in range(B):
                # Decode boxes for the current image `i`
                # We need to pass the specific anchors and deltas for this image
                # `anchors_batched[i]` and `deltas[i]` are the correct tensors to use
                decoded_boxes_norm_i = tf.numpy_function(
                    decode_boxes_from_deltas,
                    [anchors_batched[i], deltas[i]],
                    Tout=tf.float32
                )
                decoded_boxes_norm_i.set_shape((anchors_batched.shape[1], 4))
    
                # Select the top-scoring box for the current image
                best_anchor_idx = tf.argmax(scores_prob[i])
                predicted_box = decoded_boxes_norm_i[best_anchor_idx]
                
                gt_box = gt_boxes[i]
    
                iou_val = iou_box(tf.expand_dims(predicted_box, axis=0), tf.expand_dims(gt_box, axis=0))
                total_iou += iou_val[0,0]
                num_predictions += 1
                
                if step % 10 == 0 and i == 0:
                    print(f"Validation Step {step}, Image {i}: IoU = {iou_val[0,0]:.4f}")
    
        if num_predictions > 0:
            average_iou = total_iou / num_predictions
            print(f"\nAverage IoU on validation set: {average_iou:.4f}")
        else:
            print("\nNo predictions made on validation set.")

    