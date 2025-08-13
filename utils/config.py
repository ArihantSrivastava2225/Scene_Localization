import os
import json
import tensorflow as tf

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def set_seed(seed=42):
    import numpy as np
    import random
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
