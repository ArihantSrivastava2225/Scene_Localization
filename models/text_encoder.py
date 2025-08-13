import tensorflow as tf
from transformers import TFBertModel, BertTokenizer

class TransformerTextEncoder(tf.keras.Model):
    def __init__(self, pretrained_model_name='bert-base-uncased', trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.bert = TFBertModel.from_pretrained(pretrained_model_name, from_pt=True)
        self.bert.trainable = trainable  # Freeze or fine-tune BERT

    def call(self, input_ids, attention_mask=None):
        """
        input_ids: [B, T] int32 token IDs
        attention_mask: [B, T] 0/1 mask (optional)
        
        Returns:
            outputs.last_hidden_state: [B, T, H] contextual embeddings
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

# Helper function to get tokenizer
def get_tokenizer(pretrained_model_name='bert-base-uncased'):
    return BertTokenizer.from_pretrained(pretrained_model_name)
