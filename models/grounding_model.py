import tensorflow as tf
from models.image_encoder import build_image_encoder
from models.text_encoder import TransformerTextEncoder
from models.attention_scorer import CrossModalAttentionScorer
# The following line is correctly commented out as per previous corrections
# from data.anchor_utils import generate_anchor_boxes

class VisualGroundingModel(tf.keras.Model):
    def __init__(self, vocab_size, num_regions=49, anchors_per_region=5):
        super().__init__()
        self.image_encoder = build_image_encoder()
        self.text_encoder = TransformerTextEncoder(trainable=False)
        self.anchor_proj = tf.keras.layers.Dense(512)
        self.scorer = CrossModalAttentionScorer(hidden_dim=512)

        self.reg_head = tf.keras.layers.Dense(4)
        self.cls_head = tf.keras.layers.Dense(1, name="score_logits")

        self.num_regions = num_regions
        self.anchors_per_region = anchors_per_region

    def call(self, image, input_ids, attention_mask, all_anchors, training=False):
        """
        The call method now accepts the pre-computed anchors as a batched argument.
        
        Args:
            image: [B, H, W, 3] image tensor
            input_ids: [B, T] token IDs
            attention_mask: [B, T] attention mask
            all_anchors: [B, R*A, 4] pre-computed anchor box coordinates (y1, x1, y2, x2)
        """
        B = tf.shape(image)[0]

        feat_map = self.image_encoder(image, training=training)
        
        Hf = tf.shape(feat_map)[1]
        Wf = tf.shape(feat_map)[2]
        D = tf.shape(feat_map)[3]

        total_anchors = tf.shape(all_anchors)[1] # R * A
        
        # FIX: all_anchors is already batched. Use it directly for boxes_tiled
        # It needs to be reshaped from [B, R*A, 4] to [B*R*A, 4]
        boxes_tiled = tf.reshape(all_anchors, [B * total_anchors, 4])

        # Box indices for ROI Align
        box_indices = tf.repeat(tf.range(B), repeats=total_anchors)

        # ROI Align (3x3 pooling) to get features for each anchor
        anchor_feats = tf.image.crop_and_resize(
            image=feat_map,
            boxes=boxes_tiled,
            box_indices=box_indices,
            crop_size=(3, 3)
        ) # → [B*R*A, 3, 3, D]

        H_crop = 3
        W_crop = 3
        anchor_feats = tf.reshape(anchor_feats, [B, self.num_regions * self.anchors_per_region, H_crop * W_crop * D])

        anchors_proj = self.anchor_proj(anchor_feats)
        
        query_embs = self.text_encoder(input_ids, attention_mask, training=training)
        
        attended_feats = self.scorer(
            anchors_proj, 
            query_embs, 
            query_mask=attention_mask,
            training=training
        )

        # FIX: Apply the final classification head here
        scores = self.cls_head(attended_feats) # Shape: [B, R*A, 1]
        scores_reshaped = tf.squeeze(scores, axis=-1) # Shape: [B, R*A]
        
        # Apply the regression head to the same attended features
        deltas = self.reg_head(attended_feats) # Shape: [B, R*A, 4]


        # FIX: The original code returned a tiled version of 'all_anchors', which was an issue.
        # Now, `all_anchors` is already batched correctly and should be returned directly.
        return {"scores": scores_reshaped, "deltas": deltas, "anchors": all_anchors}