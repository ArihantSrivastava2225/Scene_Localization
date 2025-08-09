"""
Losses for region scoring + anchor-based grounding.

Strategy implemented here:
- For each image, compute IoU between all anchors and the GT box.
- Anchors with IoU >= pos_iou_thresh are positives (label=1)
- Anchors with IoU <= neg_iou_thresh are negatives (label=0)
- Use sigmoid BCE over all anchors (balanced by positives/negatives) for matching
- For positive anchors compute SmoothL1 loss between predicted deltas and encoded gt deltas

Note: This is a common and robust approach that mirrors detection pipelines.
"""

import tensorflow as tf
import numpy as np


def iou_boxes(boxes1, boxes2):
    """
    boxes in [N,4] and [M,4] with x1,y1,x2,y2 normalized.
    Returns IoU matrix [N, M]
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]

    boxes1 = np.expand_dims(boxes1, 1)  # [N,1,4]
    boxes2 = np.expand_dims(boxes2, 0)  # [1,M,4]

    inter_x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    union = area1 + area2 - inter_area + 1e-8
    iou = inter_area / union
    return iou


def match_anchors_to_gt(anchors_np, gt_boxes_np, pos_iou_thresh=0.5, neg_iou_thresh=0.4):
    """
    anchors_np: [A,4] numpy anchors
    gt_boxes_np: [G,4] ground-truth boxes for the image (usually G=1 here)

    Returns:
      labels: array [A] with 1 for positive, 0 for negative, -1 for ignore
      matched_gt_boxes: [A,4] where each anchor has the matched GT box (or zeros)
    """
    A = anchors_np.shape[0]
    labels = -1 * np.ones((A,), dtype=np.int32)
    matched_gt = np.zeros((A, 4), dtype=np.float32)

    if gt_boxes_np.shape[0] == 0:
        # no objects -> all negatives
        labels[:] = 0
        return labels, matched_gt

    ious = iou_boxes(anchors_np, gt_boxes_np)  # [A, G]
    best_iou = ious.max(axis=1)
    best_gt_idx = ious.argmax(axis=1)

    # positives
    labels[best_iou >= pos_iou_thresh] = 1
    # negatives
    labels[best_iou <= neg_iou_thresh] = 0

    # ensure each GT has at least one positive (find anchors with highest IoU per GT)
    gt_best_for_anchor = ious.argmax(axis=0)  # for each GT, index of anchor with max IoU
    for gt_i, a_idx in enumerate(gt_best_for_anchor):
        labels[a_idx] = 1
        best_gt_idx[a_idx] = gt_i

    # matched gt boxes per anchor
    matched_gt = gt_boxes_np[best_gt_idx]

    return labels, matched_gt


def matching_and_regression_loss(pred_scores, pred_deltas, anchors_np, gt_boxes_np,
                                 pos_iou_thresh=0.5, neg_iou_thresh=0.4,
                                 lambda_reg=1.0):
    """
    pred_scores: [A] predicted logits (before sigmoid) for anchors
    pred_deltas: [A,4] predicted deltas (tx,ty,tw,th)
    anchors_np: [A,4] anchor coords
    gt_boxes_np: [G,4] ground truth boxes for the image

    Returns: total_loss (scalar), dict of components
    """
    # match anchors
    labels, matched_gt = match_anchors_to_gt(anchors_np, gt_boxes_np,
                                              pos_iou_thresh, neg_iou_thresh)

    labels_tf = tf.convert_to_tensor(labels, dtype=tf.float32)  # 1 pos, 0 neg, -1 ignore
    # create mask for valid anchors
    valid_mask = tf.where(labels_tf >= 0.0, 1.0, 0.0)

    # classification loss (sigmoid BCE), only over valid anchors
    probs = tf.sigmoid(pred_scores)
    bce = tf.keras.losses.binary_crossentropy(labels_tf * valid_mask, probs * valid_mask)
    # bce returns per-anchor loss; mask out ignored anchors
    bce = tf.reduce_sum(bce * valid_mask) / (tf.reduce_sum(valid_mask) + 1e-6)

    # regression loss for positive anchors only
    pos_mask = tf.where(labels_tf == 1.0, 1.0, 0.0)
    num_pos = tf.reduce_sum(pos_mask)
    reg_loss = 0.0
    if tf.cast(num_pos, tf.int32) > 0:
        # prepare targets for positives
        matched_gt_tf = tf.convert_to_tensor(matched_gt, dtype=tf.float32)
        target_deltas = encode_boxes_np_tf(anchors_np, matched_gt)
        target_deltas = tf.convert_to_tensor(target_deltas, dtype=tf.float32)

        diff = (pred_deltas - target_deltas) * tf.expand_dims(pos_mask, axis=-1)
        smooth_l1 = tf.keras.losses.Huber(delta=1.0, reduction=tf.keras.losses.Reduction.SUM)(
            tf.zeros_like(diff), diff
        )
        reg_loss = smooth_l1 / (num_pos + 1e-6)
    else:
        reg_loss = tf.constant(0.0)

    total_loss = bce + lambda_reg * reg_loss
    return total_loss, {"bce": bce.numpy().item() if isinstance(bce, tf.Tensor) else float(bce),
                        "reg_loss": float(reg_loss)}


def encode_boxes_np_tf(anchors_np, matched_gt_np):
    """
    Helper to compute target deltas using numpy (kept out of TF graph). Returns [A,4]
    """
    # reuse encode_boxes function but implemented here inline to avoid circular imports
    def anchors_to_centers_np(a):
        x1, y1, x2, y2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        w = x2 - x1
        h = y2 - y1
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h
        return np.stack([cx, cy, w, h], axis=-1)

    a_cent = anchors_to_centers_np(anchors_np)
    g_cent = anchors_to_centers_np(matched_gt_np)

    ax, ay, aw, ah = a_cent[:, 0], a_cent[:, 1], a_cent[:, 2], a_cent[:, 3]
    gx, gy, gw, gh = g_cent[:, 0], g_cent[:, 1], g_cent[:, 2], g_cent[:, 3]

    eps = 1e-6
    tx = (gx - ax) / (aw + eps)
    ty = (gy - ay) / (ah + eps)
    tw = np.log((gw + eps) / (aw + eps))
    th = np.log((gh + eps) / (ah + eps))

    return np.stack([tx, ty, tw, th], axis=-1).astype(np.float32)

