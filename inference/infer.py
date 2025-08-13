# infer.py
"""
Inference script that mirrors train.py preprocessing, anchor generation and decoding.

Usage:
    from infer import infer_and_visualize
    decoded_boxes, scores = infer_and_visualize(
        model,                      # your VisualGroundingModel loaded with weights
        image_path,                 # path to image file
        query,                      # raw string query
        anchors=pre_computed_anchors, # pre-computed anchors from main.py
        # ... other optional args
    )
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from transformers import BertTokenizer

from data.anchor_utils import decode_boxes_from_deltas, generate_anchor_boxes

# --- Helper: prepare text and image as before ---
def prepare_text(tokenizer, query, max_len=20):
    if isinstance(query, str):
        enc = tokenizer(
            query,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        return enc['input_ids'].astype(np.int32), enc['attention_mask'].astype(np.int32)
    else:
        raise ValueError("Query must be a string for inference.")


def preprocess_image_for_model(image_path, image_size=(224,224)):
    img = Image.open(image_path).convert("RGB").resize(image_size)
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    return tf.convert_to_tensor(arr)[None, ...], int(img.height), int(img.width), arr


def convert_decoded_to_pixel_boxes(decoded_boxes, orig_h, orig_w):
    boxes_px = decoded_boxes.copy()
    boxes_px[:, 0] = boxes_px[:, 0] * orig_w
    boxes_px[:, 2] = boxes_px[:, 2] * orig_w
    boxes_px[:, 1] = boxes_px[:, 1] * orig_h
    boxes_px[:, 3] = boxes_px[:, 3] * orig_h
    return boxes_px


def infer_and_visualize(
        model,
        image_path,
        query,
        anchors, # FIX: Accept pre-computed anchors as a parameter
        image_size=(224,224),
        feat_h=7, feat_w=7,
        anchors_per_region=9,
        top_k=5,
        conf_thresh=0.3,
        nms_iou_thresh=0.5,
        tokenizer_name='bert-base-uncased',
        max_len=20,
        show=True
    ):
    """
    Runs inference using the same preprocessing/anchors as training and plots results.
    model: VisualGroundingModel instance (weights already loaded)
    query: raw string
    anchors: Pre-computed anchors, must be in [R*A, 4] format
    """

    # 1) Prepare text inputs
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    input_ids_np, attention_mask_np = prepare_text(tokenizer, query, max_len=max_len)
    # FIX: Remove the redundant tf.expand_dims calls
    # The prepare_text function already returns tensors with a batch dimension.
    # The shape should be (1, 20), not (1, 1, 20).
    input_ids = input_ids_np
    attention_mask = attention_mask_np

    # 2) Load and preprocess image
    img_batch, orig_h, orig_w, img_arr = preprocess_image_for_model(image_path, image_size=image_size)
    
    # 3) FIX: Pass anchors directly to the model after adding a batch dimension
    anchors_batched = tf.expand_dims(anchors, axis=0)
    
    # 4) Run model forward
    # FIX: Pass the pre-computed anchors to the model
    preds = model(img_batch, input_ids, attention_mask, anchors_batched, training=False)

    # 5) Extract scores & deltas
    # The output format is now guaranteed to be a dictionary thanks to the fix
    raw_scores = preds['scores']
    deltas = preds.get('deltas', None)

    # Convert to numpy and remove batch dim
    raw_scores_np = raw_scores.numpy().squeeze()
    scores = 1 / (1 + np.exp(-raw_scores_np)) # Sigmoid to probs

    if deltas is not None:
        deltas_np = deltas.numpy().squeeze()
    else:
        # If no regression head available, use zeros
        deltas_np = np.zeros((anchors.shape[0], 4), dtype=np.float32)

    # 6) Decode to boxes (normalized coordinates)
    decoded_boxes_norm = decode_boxes_from_deltas(anchors.numpy(), deltas_np)

    # 7) Filter by confidence threshold + NMS (on pixel coords)
    decoded_boxes_px = convert_decoded_to_pixel_boxes(decoded_boxes_norm.copy(), orig_h, orig_w)
    
    # Convert from [x1,y1,x2,y2] to [y1,x1,y2,x2] for TensorFlow NMS
    boxes_for_tf = np.stack([decoded_boxes_px[:,1], decoded_boxes_px[:,0],
                             decoded_boxes_px[:,3], decoded_boxes_px[:,2]], axis=1).astype(np.float32)

    # Apply confidence threshold
    valid_idx = np.where(scores >= conf_thresh)[0]
    
    if valid_idx.size == 0:
        print("[infer] No predictions above confidence threshold. Returning top K raw scores.")
        top_idx = np.argsort(scores)[-top_k:][::-1]
    else:
        valid_boxes = boxes_for_tf[valid_idx]
        valid_scores = scores[valid_idx].astype(np.float32)
        
        # Sort by score and take top K for NMS
        order = np.argsort(valid_scores)[-min(200, valid_idx.size):][::-1]
        
        selected = tf.image.non_max_suppression(
            boxes=tf.convert_to_tensor(valid_boxes[order], dtype=tf.float32),
            scores=tf.convert_to_tensor(valid_scores[order], dtype=tf.float32),
            max_output_size=top_k,
            iou_threshold=nms_iou_thresh,
            score_threshold=conf_thresh
        ).numpy()

        top_idx = valid_idx[order[selected]]

    # 8) Plot results on original image
    fig, ax = plt.subplots(1, figsize=(8,8))
    ax.imshow(img_arr)
    H, W = orig_h, orig_w

    for i, idx in enumerate(top_idx):
        # Your anchor_utils returns [y1, x1, y2, x2] and plots with x1, y1
        box = decoded_boxes_norm[idx]
        y1, x1, y2, x2 = box[0], box[1], box[2], box[3]
        
        x1_px = x1 * W
        y1_px = y1 * H
        w_px = (x2 - x1) * W
        h_px = (y2 - y1) * H

        rect = plt.Rectangle((x1_px, y1_px), w_px, h_px, fill=False,
                             linewidth=2 if i==0 else 1,
                             edgecolor='r' if i==0 else 'y')
        ax.add_patch(rect)
        ax.text(x1_px, max(0, y1_px-6), f"{i+1}: {scores[idx]:.3f}",
                color='white', backgroundcolor='black', fontsize=8)

    ax.axis('off')
    if show:
        plt.show()

    return decoded_boxes_norm, scores