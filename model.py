from helperFunctions import *

def build_generator(input_shape):
    # Define a convolution block with optional batch normalization and customizable activation
    def conv2d_block(input_tensor, num_filters, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2), use_norm=True):
        x = Conv2D(num_filters, kernel_size, strides=strides, padding=padding)(input_tensor)
        if use_norm:
            x = BatchNormalization()(x)
        x = activation(x)
        return x

    # Define a deconvolution (transposed convolution) block with optional batch normalization and customizable activation
    def deconv2d_block(input_tensor, num_filters, kernel_size=4, strides=2, padding='same', activation=ReLU(), use_norm=True):
        x = Conv2DTranspose(num_filters, kernel_size, strides=strides, padding=padding)(input_tensor)
        if use_norm:
            x = BatchNormalization()(x)
        x = activation(x)
        return x

    input_tensor = Input(shape=input_shape)
    x = input_tensor

    # Downsampling layers
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

    # Upsampling layers
    x = deconv2d_block(x, 256)
    x = deconv2d_block(x, 128)
    x = deconv2d_block(x, 64)

    x = Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh')(x)

    return Model(input_tensor, x)

def build_discriminator(input_shape):
    # Reusing the conv2d_block from the generator
    def conv2d_block(input_tensor, num_filters, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2), use_norm=True):
        x = Conv2D(num_filters, kernel_size, strides=strides, padding=padding)(input_tensor)
        if use_norm:
            x = BatchNormalization()(x)
        x = activation(x)
        return x

    input_tensor = Input(shape=input_shape)
    x = input_tensor

    # Discriminator layers
    x = conv2d_block(x, 64, use_norm=False)
    x = conv2d_block(x, 128)
    x = conv2d_block(x, 256)
    x = conv2d_block(x, 512, strides=1)

    x = Conv2D(1, kernel_size=4, strides=1, padding='same')(x)

    return Model(input_tensor, x)

# Define the discriminator loss function
def discriminator_loss(real_output, fake_output):
    real_loss = BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Define the generator loss function
def generator_loss(fake_output):
    return BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)

# Define the cycle consistency loss function
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

