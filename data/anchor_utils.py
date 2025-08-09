"""
Anchor generation utilities.
Generates anchor boxes for a CNN feature map grid. Anchor coordinates are
returned in normalized form (x1, y1, x2, y2) relative to image width/height.
"""

import numpy as np

def generate_anchors_for_feature_map(feat_h, feat_w, image_size=(224, 224), scales=(32, 64, 128), ratios=(0.5, 1.0, 2.0)):
    """
    Generate anchors centered on each cell of a feature map.

    Args:
        feat_h, feat_w: feature map spatial dims (e.g., 7,7)
        image_size: (H, W) of input image
        scales: list/tuple of scales (in pixels) relative to original image
        ratios: list/tuple of aspect ratios (h/w)

    Returns:
        anchors: np.array of shape [R*B, 4] where R = feat_h*feat_w, B = len(scales)*len(ratios)
                 each row = [x1, y1, x2, y2] normalized to [0,1]
    """
    img_h, img_w = image_size
    stride_h = img_h / feat_h
    stride_w = img_w / feat_w

    #centers of the cells
    centers_y = (np.arange(feat_h) + 0.5) * stride_h
    centers_x = (np.arange(feat_w) + 0.5) * stride_w
    centers = np.stack(np.meshgrid(centers_x, centers_y), axis=-1)  #[feat_h, feat_w, 2] x,y

    anchors = []
    for i in range(feat_h):
        for j in range(feat_w):
            cx = centers[i, j, 0]
            cy = centers[i, j, 1]
            for scale in scales:
                for ratio in ratios:
                    #compute box size
                    w = scale*np.sqrt(1.0/ratio)
                    h = scale*np.sqrt(ratio)

                    x1 = (cx - w / 2.0) / img_w
                    y1 = (cy - h / 2.0) / img_h
                    x2 = (cx + w / 2.0) / img_w
                    y2 = (cy + h / 2.0) / img_h

                    #clip to [0,1]
                    x1 = max(0.0, x1); y1 = max(0.0, y1)
                    x2 = min(1.0, x2); y2 = min(1.0, y2)

                    anchors.append([x1, y1, x2, y2])

    anchors = np.array(anchors, dtype=np.float32)
    return anchors 

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