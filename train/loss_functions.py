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


def matching_and_regression_loss(pred_scores, pred_deltas, anchors_tf, gt_boxes_tf,
                                  pos_iou_thresh=0.5, neg_iou_thresh=0.4, lambda_reg=5.0):
    """
    Computes classification and regression loss for a batch.
    pred_scores: [B, A] predicted logits
    pred_deltas: [B, A, 4] predicted deltas
    anchors_tf: [B, A, 4] anchor coordinates
    gt_boxes_tf: [B, G, 4] ground truth boxes
    Returns: total_loss (scalar), dict of components
    """
    # Match anchors to ground truth for the whole batch
    labels, matched_gt = match_anchors_to_gt(anchors_tf, gt_boxes_tf, pos_iou_thresh, neg_iou_thresh)

    # Classification loss
    valid_mask = tf.where(labels >= 0.0, 1.0, 0.0)
    # FIX: Use from_logits=True and pass pred_scores directly.
    # The labels tensor has shape [B, A], and pred_scores has shape [B, A].
    # They should have the same shape before applying the loss.
    # FIX: Manually compute binary cross-entropy with logits.
    # This ensures the output shape is consistent with the labels and valid_mask.
    # It also handles numerical stability with logits.
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=pred_scores)
    
    # The BCE output now has shape [B, 441], compatible with valid_mask
    bce = tf.reduce_sum(bce * valid_mask) / (tf.reduce_sum(valid_mask) + 1e-6)

    # Regression loss (only for positive anchors)
    pos_mask = tf.where(labels == 1.0, 1.0, 0.0)
    num_pos = tf.reduce_sum(pos_mask)
    reg_loss = 0.0
    if tf.cast(num_pos, tf.int32) > 0:
        # Calculate target deltas for positive anchors
        target_deltas = encode_boxes_tf(anchors_tf, matched_gt)
        
        # Apply mask to both predicted and target deltas
        pred_deltas_masked = pred_deltas * tf.expand_dims(pos_mask, axis=-1)
        target_deltas_masked = target_deltas * tf.expand_dims(pos_mask, axis=-1)
        
        # Calculate Huber loss on the masked deltas
        huber_loss_op = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        reg_loss = huber_loss_op(target_deltas_masked, pred_deltas_masked) / (num_pos + 1e-6)
    
    total_loss = bce + lambda_reg * reg_loss
    
    return total_loss, {"bce": bce, "reg_loss": reg_loss}