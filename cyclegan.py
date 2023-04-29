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

def build_generator(input_shape):
    def conv2d_block(input_tensor, num_filters, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2), use_norm=True):
        x = Conv2D(num_filters, kernel_size, strides=strides, padding=padding)(input_tensor)
        if use_norm:
            x = BatchNormalization()(x)
        x = activation(x)
        return x

    def deconv2d_block(input_tensor, num_filters, kernel_size=4, strides=2, padding='same', activation=ReLU(), use_norm=True):
        x = Conv2DTranspose(num_filters, kernel_size, strides=strides, padding=padding)(input_tensor)
        if use_norm:
            x = BatchNormalization()(x)
        x = activation(x)
        return x

    input_tensor = Input(shape=input_shape)
    x = input_tensor

    # Downsampling
    x = conv2d_block(x, 64, use_norm=False)
    x = conv2d_block(x, 128)
    x = conv2d_block(x, 256)
    x = conv2d_block(x, 512)

    # Residual blocks
    for _ in range(9):
        x_res = x
        x_res = conv2d_block(x_res, 512, strides=1)
        x_res = conv2d_block(x_res, 512, strides=1)
        x += x_res

    # Upsampling
    x = deconv2d_block(x, 256)
    x = deconv2d_block(x, 128)
    x = deconv2d_block(x, 64)

    x = Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh')(x)

    return Model(input_tensor, x)

def build_discriminator(input_shape):
    def conv2d_block(input_tensor, num_filters, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2), use_norm=True):
        x = Conv2D(num_filters, kernel_size, strides=strides, padding=padding)(input_tensor)
        if use_norm:
            x = BatchNormalization()(x)
        x = activation(x)
        return x

    input_tensor = Input(shape=input_shape)
    x = input_tensor

    x = conv2d_block(x, 64, use_norm=False)
    x = conv2d_block(x, 128)
    x = conv2d_block(x, 256)
    x = conv2d_block(x, 512, strides=1)

    x = Conv2D(1, kernel_size=4, strides=1, padding='same')(x)

    return Model(input_tensor, x)


def discriminator_loss(real_output, fake_output):
    real_loss = BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)

def cycle_consistency_loss(real_image, cycled_image, LAMBDA=10):
    loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return LAMBDA * loss

@tf.function
def train_step(real_a, real_b, generator_a2b, generator_b2a, discriminator_a, discriminator_b, gen_a2b_optimizer, gen_b2a_optimizer, disc_a_optimizer, disc_b_optimizer):
    with tf.GradientTape(persistent=True) as tape:
        # Forward pass through generators
        fake_b = generator_a2b(real_a, training=True)
        fake_a = generator_b2a(real_b, training=True)

        # Cycle back to original domain
        cycled_a = generator_b2a(fake_b, training=True)
        cycled_b = generator_a2b(fake_a, training=True)

        # Forward pass through discriminators
        disc_real_a_output = discriminator_a(real_a, training=True)
        disc_real_b_output = discriminator_b(real_b, training=True)
        disc_fake_a_output = discriminator_a(fake_a, training=True)
        disc_fake_b_output = discriminator_b(fake_b, training=True)

        # Calculate losses
        gen_a2b_loss = generator_loss(disc_fake_b_output)
        gen_b2a_loss = generator_loss(disc_fake_a_output)
        disc_a_loss = discriminator_loss(disc_real_a_output, disc_fake_a_output)
        disc_b_loss = discriminator_loss(disc_real_b_output, disc_fake_b_output)

        # Calculate cycle consistency loss
        cycle_a_loss = cycle_consistency_loss(real_a, cycled_a)
        cycle_b_loss = cycle_consistency_loss(real_b, cycled_b)

        # Calculate total generator loss
        total_gen_a2b_loss = gen_a2b_loss + cycle_a_loss + cycle_b_loss
        total_gen_b2a_loss = gen_b2a_loss + cycle_a_loss + cycle_b_loss

    # Calculate gradients
    gen_a2b_gradients = tape.gradient(total_gen_a2b_loss, generator_a2b.trainable_variables)
    gen_b2a_gradients = tape.gradient(total_gen_b2a_loss, generator_b2a.trainable_variables)
    disc_a_gradients = tape.gradient(disc_a_loss, discriminator_a.trainable_variables)
    disc_b_gradients = tape.gradient(disc_b_loss, discriminator_b.trainable_variables)

    # Apply gradients
    gen_a2b_optimizer.apply_gradients(zip(gen_a2b_gradients, generator_a2b.trainable_variables))
    gen_b2a_optimizer.apply_gradients(zip(gen_b2a_gradients, generator_b2a.trainable_variables))
    disc_a_optimizer.apply_gradients(zip(disc_a_gradients, discriminator_a.trainable_variables))
    disc_b_optimizer.apply_gradients(zip(disc_b_gradients, discriminator_b.trainable_variables))

    return total_gen_a2b_loss, total_gen_b2a_loss, disc_a_loss, disc_b_loss


def train(dataset_a, dataset_b, epochs, steps_per_epoch, checkpoint_dir):
    generator_a2b = build_generator((256, 256, 3))
    generator_b2a = build_generator((256, 256, 3))
    discriminator_a = build_discriminator((256, 256, 3))
    discriminator_b = build_discriminator((256, 256, 3))

    gen_a2b_optimizer = Adam(2e-4, beta_1=0.5)
    gen_b2a_optimizer = Adam(2e-4, beta_1=0.5)
    disc_a_optimizer = Adam(2e-4, beta_1=0.5)
    disc_b_optimizer = Adam(2e-4, beta_1=0.5)

    # Create checkpoint and checkpoint manager
    checkpoint = Checkpoint(
        epoch=tf.Variable(0),
        step=tf.Variable(0),
        generator_a2b=generator_a2b,
        generator_b2a=generator_b2a,
        discriminator_a=discriminator_a,
        discriminator_b=discriminator_b,
        gen_a2b_optimizer=gen_a2b_optimizer,
        gen_b2a_optimizer=gen_b2a_optimizer,
        disc_a_optimizer=disc_a_optimizer,
        disc_b_optimizer=disc_b_optimizer,
    )

    checkpoint_manager = CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

    # Restore the latest checkpoint
    latest_checkpoint = checkpoint_manager.latest_checkpoint
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint)
        print(f"Restored from checkpoint {latest_checkpoint}")

    step = checkpoint.step.numpy()

    for _ in range(checkpoint.epoch.numpy(), epochs):
        for real_a, real_b in tf.data.Dataset.zip((dataset_a, dataset_b)):
            step += 1
            gen_a2b_loss, gen_b2a_loss, disc_a_loss, disc_b_loss = train_step(real_a, real_b, generator_a2b, generator_b2a, discriminator_a, discriminator_b, gen_a2b_optimizer, gen_b2a_optimizer, disc_a_optimizer, disc_b_optimizer)

            checkpoint.step.assign(step)

            if step % 200 == 0:
                print(f"Epoch: {checkpoint.epoch.numpy()}, Step: {step}")
                print(f"Generator A2B Loss: {gen_a2b_loss:.4f}, Generator B2A Loss: {gen_b2a_loss:.4f}")
                print(f"Discriminator A Loss: {disc_a_loss:.4f}, Discriminator B Loss: {disc_b_loss:.4f}")

            if step % steps_per_epoch == 0:
                break

        checkpoint.epoch.assign_add(1)

        # Save checkpoint
        checkpoint_manager.save()
        print(f"Saved checkpoint at step {step}")



def save_image(image_array, save_path):
    image = (image_array + 1) * 127.5
    image = Image.fromarray(image.astype(np.uint8))
    image.save(save_path)



def test(input_dataset, generator_a2b_path, generator_b2a_path, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load pre-trained generator models
    generator_a2b = tf.keras.models.load_model(generator_a2b_path)
    generator_b2a = tf.keras.models.load_model(generator_b2a_path)

    # Test the generators on the input dataset
    for i, image in enumerate(input_dataset):
        generated_b = generator_a2b(image[np.newaxis, ...], training=False)
        generated_a = generator_b2a(image[np.newaxis, ...], training=False)

        # Save the generated images
        save_image(generated_b[0].numpy(), os.path.join(output_dir, f'generated_b_{i}.jpg'))
        save_image(generated_a[0].numpy(), os.path.join(output_dir, f'generated_a_{i}.jpg'))




def load_image(image_path, img_height=256, img_width=256):
    # Load and preprocess the image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, [img_height, img_width])
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    return image

def load_dataset(dataset_path, batch_size, img_height=256, img_width=256):
    # Load and preprocess the dataset
    dataset = tf.data.Dataset.list_files(dataset_path + '/*.jpg', shuffle=True)
    dataset = dataset.map(lambda x: load_image(x, img_height, img_width))
    dataset = dataset.batch(batch_size)
    return dataset


apples_dataset = load_dataset("apple2oranges/apples/train", batch_size=1)
oranges_dataset = load_dataset("apple2oranges/oranges/train", batch_size=1)

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

steps_per_epoch = 200  # Adjust this value according to the size of your datasets
train(apples_dataset, oranges_dataset, epochs=200, steps_per_epoch=steps_per_epoch, checkpoint_dir=checkpoint_dir)


# Load test datasets
apples_test_dataset = load_dataset("apple2oranges/apples/test", batch_size=1)
oranges_test_dataset = load_dataset("apple2oranges/oranges/test", batch_size=1)

# Test the model using the latest saved generator models
test(apples_test_dataset, "generator_a2b_epoch_199.h5", "generator_b2a_epoch_199.h5", "test_results/apples")
test(oranges_test_dataset, "generator_a2b_epoch_199.h5", "generator_b2a_epoch_199.h5", "test_results/oranges")
