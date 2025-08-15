"""
Anchor generation utilities.
Generates anchor boxes for a CNN feature map grid. Anchor coordinates are
returned in normalized form (x1, y1, x2, y2) relative to image width/height.
"""

import numpy as np
import tensorflow as tf

def generate_anchor_boxes(Hf, Wf, anchors_per_region, 
                           image_height=224, image_width=224,
                           scales=[0.5, 1.0, 2.0], aspect_ratios=[0.5, 1.0, 2.0]):
    """
    Generate normalized anchor boxes for an Hf x Wf feature map.

    Returns:
        boxes: [Hf*Wf*anchors_per_region, 4] in normalized coords (ymin, xmin, ymax, xmax)
    """
    # Step size in original image
    stride_y = image_height / Hf
    stride_x = image_width / Wf

    # Centers of each cell in original image coords
    cy = np.arange(stride_y/2, image_height, stride_y)
    cx = np.arange(stride_x/2, image_width, stride_x)

    centers = [(x, y) for y in cy for x in cx]  # (x_center, y_center)

    anchors = []
    for (x_center, y_center) in centers:
        for scale in scales:
            for ratio in aspect_ratios:
                # Compute anchor box height/width
                h = scale * stride_y * np.sqrt(ratio)
                w = scale * stride_x / np.sqrt(ratio)

                ymin = (y_center - h/2) / image_height
                xmin = (x_center - w/2) / image_width
                ymax = (y_center + h/2) / image_height
                xmax = (x_center + w/2) / image_width

                anchors.append([ymin, xmin, ymax, xmax])

    anchors = np.clip(anchors, 0, 1)  # ensure within image bounds
    return tf.constant(anchors, dtype=tf.float32)

def anchors_to_centers(anchors):
    """Convert [x1,y1,x2,y2] to center format [cx,cy,w,h] (normalized)."""
    x1, y1, x2, y2 = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    w = x2 - x1
    h = y2 - y1
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return np.stack([cx, cy, w, h], axis=-1)

def encode_boxes(anchors, gt_boxes):
    """
    Encode ground-truth boxes relative to anchors using standard box deltas.
    anchors, gt_boxes are arrays in normalized [x1,y1,x2,y2] format.

    Returns deltas [tx, ty, tw, th] for each matched anchor.
    """
    # anchors and gt_boxes are [N,4]
    a_cent = anchors_to_centers(anchors)
    g_cent = anchors_to_centers(gt_boxes)

    ax, ay, aw, ah = a_cent[:, 0], a_cent[:, 1], a_cent[:, 2], a_cent[:, 3]
    gx, gy, gw, gh = g_cent[:, 0], g_cent[:, 1], g_cent[:, 2], g_cent[:, 3]

    eps = 1e-6
    tx = (gx - ax) / (aw + eps)
    ty = (gy - ay) / (ah + eps)
    tw = np.log((gw + eps) / (aw + eps))
    th = np.log((gh + eps) / (ah + eps))

    return np.stack([tx, ty, tw, th], axis=-1)

def decode_boxes_from_deltas(anchors, deltas):
    """
    Decode predicted deltas back to [x1,y1,x2,y2] normalized coords.
    anchors: [N,4], deltas: [N,4]
    """
    a_cent = anchors_to_centers(anchors)
    ax, ay, aw, ah = a_cent[:, 0], a_cent[:, 1], a_cent[:, 2], a_cent[:, 3]
    tx, ty, tw, th = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]

    gx = tx * aw + ax
    gy = ty * ah + ay
    gw = np.exp(tw) * aw
    gh = np.exp(th) * ah

    x1 = gx - 0.5 * gw
    y1 = gy - 0.5 * gh
    x2 = gx + 0.5 * gw
    y2 = gy + 0.5 * gh

    boxes = np.stack([x1, y1, x2, y2], axis=-1)
    boxes[:, 0] = np.clip(boxes[:, 0], 0.0, 1.0)
    boxes[:, 1] = np.clip(boxes[:, 1], 0.0, 1.0)
    boxes[:, 2] = np.clip(boxes[:, 2], 0.0, 1.0)
    boxes[:, 3] = np.clip(boxes[:, 3], 0.0, 1.0)
    return boxes

def cxcywh_to_xyxy(boxes):
    # boxes: [..., 4] -> (cx, cy, w, h) to (x1, y1, x2, y2)
    cx, cy, w, h = tf.split(boxes, 4, axis=-1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return tf.concat([x1, y1, x2, y2], axis=-1)

