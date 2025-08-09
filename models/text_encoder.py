import tensorflow as tf

def build_text_encoder(vocab_size, embedding_dims=300, lstm_units=512):
    inputs = tf.keras.Input(shape=(None,))
    x = tf.keras.layers.Embedding(vocab_size, embedding_dims)(inputs)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True))(x)
    return tf.keras.Model(inputs, x, name="text-encoder")
