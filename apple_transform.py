import tensorflow as tf
import tensorflow_transform as tft

# Fitur numerik dari dataset Apple Quality
NUMERIC_FEATURES = [
    'A_id', 'Size', 'Weight', 'Sweetness', 
    'Crunchiness', 'Juiciness', 'Ripeness', 
    'Acidity'
]
LABEL_KEY = 'Quality'

def transformed_name(key):
    return f"{key}_xf"

def preprocessing_fn(inputs):
    """Preprocessing function sederhana."""
    outputs = {}
    
    # Normalisasi fitur numerik
    for feature in NUMERIC_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_z_score(inputs[feature])
    
    # Konversi label ke binary (0/1)
    outputs[transformed_name(LABEL_KEY)] = tf.cast(
        tf.equal(inputs[LABEL_KEY], 'good'), 
        tf.int64
    )
    
    return outputs