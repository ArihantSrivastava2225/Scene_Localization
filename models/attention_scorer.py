import tensorflow as tf

class CrossModalAttentionScorer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim=512, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.region_proj = tf.keras.layers.Dense(hidden_dim, use_bias=False, name="region_proj")
        self.query_proj = tf.keras.layers.Dense(hidden_dim, use_bias=False, name="query_proj")

        # FIX: Replaced single dense layer with a two-layer MLP with ReLU activation
        self.mlp_1 = tf.keras.layers.Dense(hidden_dim, activation="relu", name="mlp_1")
        self.mlp_2 = tf.keras.layers.Dense(hidden_dim, name="mlp_2")
        self.dropout_layer = tf.keras.layers.Dropout(dropout) if dropout > 0 else None
        
    def call(self, anchor_feats, query_embs, query_mask=None, training=False):
        # FIX: Normalize features before projection to focus on direction
        R_proj = self.region_proj(tf.nn.l2_normalize(anchor_feats, axis=-1))
        Q_proj = self.query_proj(tf.nn.l2_normalize(query_embs, axis=-1))
        
        # Calculate scores as a dot product
        scores_raw = tf.einsum('bah,bth->bat', R_proj, Q_proj)

        if query_mask is not None:
            mask = tf.cast(tf.expand_dims(query_mask, axis=1), dtype=scores_raw.dtype)
            very_negative = tf.constant(-1e9, dtype=scores_raw.dtype)
            scores_raw = tf.where(mask > 0, scores_raw, very_negative)

        attn_weights = tf.nn.softmax(scores_raw, axis=-1)
        attended_queries = tf.einsum('bat,bth->bah', attn_weights, Q_proj)

        combined = tf.concat([anchor_feats, attended_queries, anchor_feats * attended_queries], axis=-1)
        
        # FIX: Pass combined features through the MLP
        x = self.mlp_1(combined)
        x = self.mlp_2(x)
        if self.dropout_layer is not None:
            x = self.dropout_layer(x, training=training)

        # FIX: Return the combined feature vector and the raw dot-product scores for contrastive loss
        return x, scores_raw

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout
        })
        return cfg