import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

def get_preprocess_function(model_name):
    if model_name == 'inceptionv3':
        return inception_preprocess
    elif model_name == 'densenet121':
        return densenet_preprocess
    return None

def get_augmentation_layer(image_size):
    return tf.keras.Sequential([
        tf.keras.layers.Resizing(image_size, image_size),
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.2),
    ])