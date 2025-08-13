import tensorflow as tf

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load image and apply the same preprocessing as in training."""
    img_raw = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img_raw, channels=3)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32) / 255.0  # normalize [0,1]
    return img
