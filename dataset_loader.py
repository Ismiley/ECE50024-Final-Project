# dataset_loader.py
import tensorflow as tf

def load_image(file):
    image = tf.io.read_file(file)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [256, 256])
    return image

def load_dataset(path, batch_size, split='train', split_ratio=0.2, shuffle=True):
    dataset = tf.data.Dataset.list_files(path + "/*.jpg", shuffle=shuffle)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset_size = len(dataset)
    val_size = int(dataset_size * split_ratio)
    train_size = dataset_size - val_size

    if split == 'train':
        return dataset.take(train_size).batch(batch_size).repeat()
    elif split == 'val':
        return dataset.skip(train_size).batch(batch_size).repeat()
    else:
        raise ValueError("Invalid split value. Expected 'train' or 'val'.")
