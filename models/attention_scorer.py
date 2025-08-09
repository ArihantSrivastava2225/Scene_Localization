import tensorflow as tf

'''
Like an self attention model where tf.einsum plays the role of QK^T and query_embs plays V.
'''
class CrossModalAttentionScorer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query_proj = tf.keras.layers.Dense(hidden_dim)
        self.region_proj = tf.keras.layers.Dense(hidden_dim)
        self.scorer = tf.keras.layers.Dense(1)

        def call(self, region_feats, query_embs):
            #region_feats: [B, R, D], query_embs: [B, T, D]
            # B -> batch size, R-> number of regions, T-> number of tokens in a query, D-> feature dimension
            R_proj = self.region_proj(region_feats)  #[B, R, D]
            Q_proj = self.query_proj(query_embs)  #[B, T, D]

            #computes attention scores: [B, R, T]
            scores = tf.einsum('brd,btd->brt', R_proj, Q_proj)
            attn_weights = tf.nn.softmax(scores, axis=-1)  #over T
            
            #weighted sum of query embeddings: [B, R, D]
            attended_queries = tf.einsum('brt,btd->brd', attn_weights, query_embs)

            #combining region and attented query
            combined = tf.concat([region_feats, attended_queries, region_feats*attended_queries], axis=-1)
            return tf.squeeze(self.scorer(combined), axis=-1)  #[B, R]
        

