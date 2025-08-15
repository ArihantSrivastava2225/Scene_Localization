import tensorflow as tf

def build_image_encoder(output_dim=512, trainable=False):
    """
    Builds the image encoder model.
    Args:
        output_dim: The output dimension of the final Conv2D layer.
        trainable_base: Whether the ResNet50 base model's weights should be trainable.
    """
    base_model = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet', input_shape=(224, 224, 3)
    )
    print(trainable)
    # FIX: Set the trainable flag based on the function's argument
    base_model.trainable = trainable
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Conv2D(output_dim, 1, activation='relu')
    ])
    
    return model