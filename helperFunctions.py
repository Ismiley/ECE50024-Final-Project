import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ReLU, LeakyReLU, BatchNormalization, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.train import Checkpoint, CheckpointManager

# Save the image
def save_image(image_array, save_path):
    image = (image_array + 1) * 127.5
    image = Image.fromarray(image.astype(np.uint8))
    image.save(save_path)

# Load and preprocess the image
def load_image(image_path, img_height=256, img_width=256):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, [img_height, img_width])
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    return image

# Load and preprocess the dataset
def load_dataset(dataset_path, batch_size, img_height=256, img_width=256):
    dataset = tf.data.Dataset.list_files(dataset_path + '/*.jpg', shuffle=True)
    dataset = dataset.map(lambda x: load_image(x, img_height, img_width))
    dataset = dataset.batch(batch_size)
    return dataset

