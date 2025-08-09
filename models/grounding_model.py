import tensorflow as tf
from models.image_encoder import build_image_encoder
from models.text_encoder import build_text_encoder
from models.attention_scorer import CrossModalAttentionScorer

class VisualGroundingModel(tf.keras.Model):
    def __init__(self, vocab_size, num_regions=49, anchors_per_region=5):
        super().__init__()
        self.image_encoder = build_image_encoder()
        self.text_encoder = build_text_encoder(vocab_size)
        self.scorer = CrossModalAttentionScorer(hidden_dim=512)
        self.num_regions = num_regions
        self.anchors_per_region = anchors_per_region

    def call(self, image, query_tokens): 
        B = tf.shape(image)[0]

        #CNN feature map -> [B, 7, 7, D] -> [B, 47, D]
        image_features = self.image_encoder(image)
        regions = tf.reshape(image_features, [B, self.num_regions, -1])

        #Query encoder -> [B, T, D]
        query_embs = self.text_encoder(query_tokens)

        #Score each region
        scores = self.scorer(regions, query_embs) #[B, R]

        return scores  #during training to calculate loss