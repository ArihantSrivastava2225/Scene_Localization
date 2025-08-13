import tensorflow as tf

class CrossModalAttentionScorer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim=512, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.region_proj = tf.keras.layers.Dense(hidden_dim, use_bias=False, name="region_proj")
        self.query_proj = tf.keras.layers.Dense(hidden_dim, use_bias=False, name="query_proj")

        self.combine_fc = tf.keras.layers.Dense(hidden_dim, activation="relu", name="combine_fc")
        self.dropout_layer = tf.keras.layers.Dropout(dropout) if dropout > 0 else None
        
        # FIX: Remove the Dense(1) layer here. The scorer should return features, not scores.
        # self.scorer = tf.keras.layers.Dense(1, name="score_logits")

    def call(self, anchor_feats, query_embs, query_mask=None, training=False):
        B, total_anchors, D_r = tf.shape(anchor_feats)[0], tf.shape(anchor_feats)[1], tf.shape(anchor_feats)[2]
        
        R_proj = self.region_proj(anchor_feats)
        Q_proj = self.query_proj(query_embs)
        scores = tf.einsum('bah,bth->bat', R_proj, Q_proj)

        if query_mask is not None:
            mask = tf.cast(tf.expand_dims(query_mask, axis=1), dtype=scores.dtype)
            very_negative = tf.constant(-1e9, dtype=scores.dtype)
            scores = tf.where(mask > 0, scores, very_negative)

        attn_weights = tf.nn.softmax(scores, axis=-1)
        attended_queries = tf.einsum('bat,bth->bah', attn_weights, Q_proj)

        combined = tf.concat([anchor_feats, attended_queries, anchor_feats * attended_queries], axis=-1)
        
        x = self.combine_fc(combined)
        if self.dropout_layer is not None:
            x = self.dropout_layer(x, training=training)

        # FIX: Return the combined feature vector 'x', not the final scores.
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout
        })
        return cfg