import tensorflow as tf

def build_image_encoder(output_dim=512):
    base_model = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet', input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Conv2D(output_dim, 1, activation='relu')  # Reduce channels
    ])
    return model
