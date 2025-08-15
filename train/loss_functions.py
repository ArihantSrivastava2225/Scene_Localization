import tensorflow as tf

def iou_boxes(boxes1, boxes2):
    """
    Computes Intersection-over-Union (IoU) for a batch of boxes.
    boxes1: [B, N, 4] and boxes2: [B, M, 4] with normalized coords [y1, x1, y2, x2].
    Returns IoU matrix: [B, N, M]
    """
    # Expand dims for broadcasting
    boxes1 = tf.expand_dims(boxes1, axis=2)  # [B, N, 1, 4]
    boxes2 = tf.expand_dims(boxes2, axis=1)  # [B, 1, M, 4]
    
    inter_x1 = tf.maximum(boxes1[..., 1], boxes2[..., 1])
    inter_y1 = tf.maximum(boxes1[..., 0], boxes2[..., 0])
    inter_x2 = tf.minimum(boxes1[..., 3], boxes2[..., 3])
    inter_y2 = tf.minimum(boxes1[..., 2], boxes2[..., 2])
    
    inter_w = tf.maximum(0.0, inter_x2 - inter_x1)
    inter_h = tf.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    area1 = (boxes1[..., 3] - boxes1[..., 1]) * (boxes1[..., 2] - boxes1[..., 0])
    area2 = (boxes2[..., 3] - boxes2[..., 1]) * (boxes2[..., 2] - boxes2[..., 0])
    
    union = area1 + area2 - inter_area + 1e-8
    iou = inter_area / union
    
    return iou

# train/loss_functions.py

import tensorflow as tf
import numpy as np


def iou_box(box1, box2):
    """
    Computes Intersection-over-Union (IoU) between two sets of boxes.
    boxes1: [N, 4] and boxes2: [M, 4] with normalized coords [y1, x1, y2, x2].
    Returns IoU matrix: [N, M]
    """
    # FIX: Expand dims for broadcasting only for the core calculation
    boxes1_exp = tf.expand_dims(box1, axis=1)  # [N, 1, 4]
    boxes2_exp = tf.expand_dims(box2, axis=0)  # [1, M, 4]
    
    # Coordinates of the intersection box
    inter_x1 = tf.maximum(boxes1_exp[..., 1], boxes2_exp[..., 1])
    inter_y1 = tf.maximum(boxes1_exp[..., 0], boxes2_exp[..., 0])
    inter_x2 = tf.minimum(boxes1_exp[..., 3], boxes2_exp[..., 3])
    inter_y2 = tf.minimum(boxes1_exp[..., 2], boxes2_exp[..., 2])
    
    inter_w = tf.maximum(0.0, inter_x2 - inter_x1)
    inter_h = tf.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    # Area of boxes1 and boxes2
    area1 = (boxes1_exp[..., 3] - boxes1_exp[..., 1]) * (boxes1_exp[..., 2] - boxes1_exp[..., 0])
    area2 = (boxes2_exp[..., 3] - boxes2_exp[..., 1]) * (boxes2_exp[..., 2] - boxes2_exp[..., 0])
    
    union = area1 + area2 - inter_area + 1e-8
    iou = inter_area / union
    
    return iou

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss for binary classification.
    y_true: [B, A]
    y_pred: [B, A] logits (raw scores)
    """
    y_pred = tf.nn.sigmoid(y_pred)
    pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1 - y_pred)
    focal_weight = tf.where(tf.equal(y_true, 1.0), alpha, 1 - alpha)
    focal_loss = -focal_weight * (1 - pt) ** gamma * tf.math.log(pt + 1e-6)
    return focal_loss


def contrastive_loss(raw_scores, labels, margin=0.5):
    """
    Contrastive loss to push positive and negative pairs apart.
    raw_scores: [B, A, T] raw dot product scores
    labels: [B, A] 1 for positive, 0 for negative, -1 for ignored
    """
    # FIX: Need to compute a single score per anchor to match the labels
    scores_per_anchor = tf.reduce_max(raw_scores, axis=-1)

    pos_mask = tf.where(labels == 1.0, 1.0, 0.0)
    neg_mask = tf.where(labels == 0.0, 1.0, 0.0)
    
    pos_scores = tf.boolean_mask(scores_per_anchor, tf.cast(pos_mask, tf.bool))
    neg_scores = tf.boolean_mask(scores_per_anchor, tf.cast(neg_mask, tf.bool))
    
    # Minimize score for positive pairs
    pos_loss = tf.reduce_mean(tf.maximum(0.0, 1.0 - pos_scores)) if tf.size(pos_scores) > 0 else 0.0
    
    # Maximize score for negative pairs, up to the margin
    neg_loss = tf.reduce_mean(tf.maximum(0.0, neg_scores - margin)) if tf.size(neg_scores) > 0 else 0.0
    
    return pos_loss + neg_loss



def match_anchors_to_gt(anchors_tf, gt_boxes_tf, pos_iou_thresh=0.5, neg_iou_thresh=0.4):
    """
    Matches anchors to ground truth boxes for a batch.
    anchors_tf: [B, A, 4] anchors (normalized)
    gt_boxes_tf: [B, G, 4] ground-truth boxes
    Returns:
      labels: [B, A] with 1 for positive, 0 for negative, -1 for ignore
      matched_gt_boxes: [B, A, 4] where each anchor has its matched GT box
    """
    B, A = tf.shape(anchors_tf)[0], tf.shape(anchors_tf)[1]
    
    # Pad gt_boxes to handle cases with no GTs (e.g., G=0)
    G = tf.shape(gt_boxes_tf)[1]
    gt_boxes_padded = tf.pad(gt_boxes_tf, [[0, 0], [0, 1], [0, 0]])
    G_padded = tf.shape(gt_boxes_padded)[1]
    
    # Calculate IoU for the whole batch
    ious = iou_boxes(anchors_tf, gt_boxes_padded)  # [B, A, G_padded]
    
    # Find best IoU for each anchor
    best_iou_per_anchor = tf.reduce_max(ious, axis=-1)  # [B, A]
    best_gt_idx_per_anchor = tf.cast(tf.argmax(ious, axis=-1), dtype=tf.int32) # [B, A]
    
    # Initialize labels
    labels = -tf.ones(tf.stack([B, A]), dtype=tf.int32)
    
    # Assign positives based on best IoU > threshold
    pos_mask = best_iou_per_anchor >= pos_iou_thresh
    labels = tf.where(pos_mask, 1, labels)

    # Assign negatives based on IoU < threshold
    neg_mask = best_iou_per_anchor < neg_iou_thresh
    labels = tf.where(neg_mask, 0, labels)
    
    # Ensure each GT has at least one positive match
    best_anchor_idx_per_gt = tf.cast(tf.argmax(ious, axis=1), dtype=tf.int32) # [B, G_padded]
    batch_indices = tf.range(B, dtype=tf.int32)
    
    # For each batch item, get the best anchor index for its one GT box (G=1)
    if G > 0:
        indices = tf.stack([batch_indices, best_anchor_idx_per_gt[:, 0]], axis=1) # [B, 2]
        labels = tf.tensor_scatter_nd_update(labels, indices, tf.ones(B, dtype=tf.int32))
    
    # Match each anchor to the corresponding GT box
    matched_gt_boxes = tf.gather(gt_boxes_padded, best_gt_idx_per_anchor, axis=1, batch_dims=1)
    
    return tf.cast(labels, dtype=tf.float32), matched_gt_boxes


def encode_boxes_tf(anchors_tf, matched_gt_tf):
    """
    Encode ground-truth boxes relative to anchors using standard box deltas.
    anchors_tf, matched_gt_tf: [B, A, 4] normalized [y1, x1, y2, x2] coords.
    Returns: deltas [B, A, 4]
    """
    eps = 1e-6

    # Convert to center format [cy, cx, h, w]
    a_cy = (anchors_tf[..., 0] + anchors_tf[..., 2]) / 2.0
    a_cx = (anchors_tf[..., 1] + anchors_tf[..., 3]) / 2.0
    a_h = anchors_tf[..., 2] - anchors_tf[..., 0]
    a_w = anchors_tf[..., 3] - anchors_tf[..., 1]
    
    g_cy = (matched_gt_tf[..., 0] + matched_gt_tf[..., 2]) / 2.0
    g_cx = (matched_gt_tf[..., 1] + matched_gt_tf[..., 3]) / 2.0
    g_h = matched_gt_tf[..., 2] - matched_gt_tf[..., 0]
    g_w = matched_gt_tf[..., 3] - matched_gt_tf[..., 1]

    # Calculate deltas
    ty = (g_cy - a_cy) / (a_h + eps)
    tx = (g_cx - a_cx) / (a_w + eps)
    th = tf.math.log((g_h + eps) / (a_h + eps))
    tw = tf.math.log((g_w + eps) / (a_w + eps))

    # Stack and return
    deltas = tf.stack([ty, tx, th, tw], axis=-1)
    return deltas


import tensorflow as tf
import numpy as np

def iou_boxes(boxes1, boxes2):
    boxes1_exp = tf.expand_dims(boxes1, axis=1)
    boxes2_exp = tf.expand_dims(boxes2, axis=0)
    inter_x1 = tf.maximum(boxes1_exp[..., 1], boxes2_exp[..., 1])
    inter_y1 = tf.maximum(boxes1_exp[..., 0], boxes2_exp[..., 0])
    inter_x2 = tf.minimum(boxes1_exp[..., 3], boxes2_exp[..., 3])
    inter_y2 = tf.minimum(boxes1_exp[..., 2], boxes2_exp[..., 2])
    inter_w = tf.maximum(0.0, inter_x2 - inter_x1)
    inter_h = tf.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area1 = (boxes1_exp[..., 3] - boxes1_exp[..., 1]) * (boxes1_exp[..., 2] - boxes1_exp[..., 0])
    area2 = (boxes2_exp[..., 3] - boxes2_exp[..., 1]) * (boxes2_exp[..., 2] - boxes2_exp[..., 0])
    union = area1 + area2 - inter_area + 1e-8
    iou = inter_area / union
    return iou


def match_anchors_to_gt(anchors_tf, gt_boxes_tf, pos_iou_thresh=0.5, neg_iou_thresh=0.4):
    B, A = tf.shape(anchors_tf)[0], tf.shape(anchors_tf)[1]
    gt_boxes_padded = tf.pad(gt_boxes_tf, [[0, 0], [0, 1], [0, 0]])
    G_padded = tf.shape(gt_boxes_padded)[1]
    ious = iou_boxes(anchors_tf, gt_boxes_padded)
    best_iou_per_anchor = tf.reduce_max(ious, axis=-1)
    best_gt_idx_per_anchor = tf.cast(tf.argmax(ious, axis=-1), dtype=tf.int32)
    labels = -tf.ones(tf.stack([B, A]), dtype=tf.int32)
    pos_mask = best_iou_per_anchor >= pos_iou_thresh
    labels = tf.where(pos_mask, 1, labels)
    neg_mask = best_iou_per_anchor < neg_iou_thresh
    labels = tf.where(neg_mask, 0, labels)
    gt_best_for_anchor = tf.cast(tf.argmax(ious, axis=-2), dtype=tf.int32)
    batch_indices = tf.reshape(tf.range(B), [B, 1])
    gt_indices = tf.range(G_padded)
    gt_best_anchor_indices = tf.stack([batch_indices, gt_best_for_anchor], axis=-1)
    updates = tf.ones(tf.shape(gt_best_anchor_indices)[0], dtype=tf.int32)
    labels = tf.tensor_scatter_nd_update(labels, gt_best_anchor_indices, updates)
    matched_gt_boxes = tf.gather(gt_boxes_padded, best_gt_idx_per_anchor, axis=1, batch_dims=1)
    return tf.cast(labels, dtype=tf.float32), matched_gt_boxes


def encode_boxes_tf(anchors_tf, matched_gt_tf):
    eps = 1e-6
    a_cy = (anchors_tf[..., 0] + anchors_tf[..., 2]) / 2.0
    a_cx = (anchors_tf[..., 1] + anchors_tf[..., 3]) / 2.0
    a_h = anchors_tf[..., 2] - anchors_tf[..., 0]
    a_w = anchors_tf[..., 3] - anchors_tf[..., 1]
    g_cy = (matched_gt_tf[..., 0] + matched_gt_tf[..., 2]) / 2.0
    g_cx = (matched_gt_tf[..., 1] + matched_gt_tf[..., 3]) / 2.0
    g_h = matched_gt_tf[..., 2] - matched_gt_tf[..., 0]
    g_w = matched_gt_tf[..., 3] - matched_gt_tf[..., 1]
    ty = (g_cy - a_cy) / (a_h + eps)
    tx = (g_cx - a_cx) / (a_w + eps)
    th = tf.math.log((g_h + eps) / (a_h + eps))
    tw = tf.math.log((g_w + eps) / (a_w + eps))
    deltas = tf.stack([ty, tx, th, tw], axis=-1)
    return deltas

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss for binary classification.
    y_true: [B, A]
    y_pred: [B, A] logits (raw scores)
    """
    y_pred = tf.nn.sigmoid(y_pred)
    pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1 - y_pred)
    focal_weight = tf.where(tf.equal(y_true, 1.0), alpha, 1 - alpha)
    focal_loss = -focal_weight * (1 - pt) ** gamma * tf.math.log(pt + 1e-6)
    return focal_loss


def contrastive_loss(raw_scores, labels, margin=0.5):
    """
    Contrastive loss to push positive and negative pairs apart.
    raw_scores: [B, A, T] raw dot product scores
    labels: [B, A] 1 for positive, 0 for negative, -1 for ignored
    """
    # FIX: Need to compute a single score per anchor to match the labels
    scores_per_anchor = tf.reduce_max(raw_scores, axis=-1)

    pos_mask = tf.where(labels == 1.0, 1.0, 0.0)
    neg_mask = tf.where(labels == 0.0, 1.0, 0.0)
    
    pos_scores = tf.boolean_mask(scores_per_anchor, tf.cast(pos_mask, tf.bool))
    neg_scores = tf.boolean_mask(scores_per_anchor, tf.cast(neg_mask, tf.bool))
    
    # Minimize score for positive pairs
    pos_loss = tf.reduce_mean(tf.maximum(0.0, 1.0 - pos_scores)) if tf.size(pos_scores) > 0 else 0.0
    
    # Maximize score for negative pairs, up to the margin
    neg_loss = tf.reduce_mean(tf.maximum(0.0, neg_scores - margin)) if tf.size(neg_scores) > 0 else 0.0
    
    return pos_loss + neg_loss


def matching_and_regression_loss(pred_scores, pred_deltas, anchors_tf, gt_boxes_tf, raw_scores,
                                  pos_iou_thresh=0.5, neg_iou_thresh=0.4,
                                  lambda_reg=1.0, lambda_contrastive=0.1, lambda_focal=1.0):
    labels, matched_gt = match_anchors_to_gt(anchors_tf, gt_boxes_tf, pos_iou_thresh, neg_iou_thresh)

    # FIX: Use focal loss instead of BCE
    valid_mask = tf.where(labels >= 0.0, 1.0, 0.0)
    focal_loss_per_anchor = focal_loss(labels, pred_scores)
    focal_loss_val = tf.reduce_sum(focal_loss_per_anchor * valid_mask) / (tf.reduce_sum(valid_mask) + 1e-6)

    # Regression Loss
    pos_mask = tf.where(labels == 1.0, 1.0, 0.0)
    num_pos = tf.reduce_sum(pos_mask)
    reg_loss = 0.0
    if tf.cast(num_pos, tf.int32) > 0:
        target_deltas = encode_boxes_tf(anchors_tf, matched_gt)
        pred_deltas_masked = pred_deltas * tf.expand_dims(pos_mask, axis=-1)
        target_deltas_masked = target_deltas * tf.expand_dims(pos_mask, axis=-1)
        huber_loss_op = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        reg_loss = huber_loss_op(target_deltas_masked, pred_deltas_masked) / (num_pos + 1e-6)
    
    # FIX: Incorporate the contrastive loss
    con_loss = contrastive_loss(raw_scores, labels)
    
    total_loss = lambda_focal * focal_loss_val + lambda_reg * reg_loss + lambda_contrastive * con_loss
    
    return total_loss, {"focal": focal_loss_val, "reg_loss": reg_loss, "contrastive": con_loss}