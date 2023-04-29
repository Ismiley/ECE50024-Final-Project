from model import *
from contextlib import redirect_stdout
import json
# Import load_dataset from dataset_loader.py
#from dataset_loader import load_dataset

def train(dataset_a, dataset_b, val_dataset_a, val_dataset_b, epochs, steps_per_epoch, checkpoint_dir):
    # Build generators and discriminators for CycleGAN
    generator_a2b = build_generator((256, 256, 3))
    generator_b2a = build_generator((256, 256, 3))
    discriminator_a = build_discriminator((256, 256, 3))
    discriminator_b = build_discriminator((256, 256, 3))

    # Set up optimizers with learning rate and momentum for generators and discriminators
    gen_a2b_optimizer = Adam(2e-4, beta_1=0.5)
    gen_b2a_optimizer = Adam(2e-4, beta_1=0.5)
    disc_a_optimizer = Adam(2e-4, beta_1=0.5)
    disc_b_optimizer = Adam(2e-4, beta_1=0.5)

    # Create a checkpoint object to save and restore model weights and optimizers
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
    '''
    def save_model_summary(model, filename):
        with open(filename, 'w') as f:
            with redirect_stdout(f):
                model.summary()

    # Save model summaries
    save_model_summary(generator_a2b, 'generator_a2b_summary.txt')
    save_model_summary(generator_b2a, 'generator_b2a_summary.txt')

    # Print model summaries
    print("Generator A2B Model Summary:")
    generator_a2b.summary()
    print("\nGenerator B2A Model Summary:")
    generator_b2a.summary()
    '''

    # Load previously saved validation accuracies if resuming training
    val_accuracies_path = os.path.join(checkpoint_dir, "val_accuracies.json")
    val_accuracies = []

    if os.path.exists(val_accuracies_path):
        try:
            with open(val_accuracies_path, "r") as f:
                val_accuracies = json.load(f)
        except json.JSONDecodeError:
            print("Error decoding JSON file. Starting with an empty list for validation accuracies.")

    # Create a checkpoint manager to handle saving and loading checkpoints
    checkpoint_manager = CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

    # Restore the latest checkpoint if it exists
    latest_checkpoint = checkpoint_manager.latest_checkpoint
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint)
        print(f"Restored from checkpoint {latest_checkpoint}")

    step = checkpoint.step.numpy()

    # Train the CycleGAN model
    for _ in range(checkpoint.epoch.numpy(), epochs):
        for real_a, real_b in tf.data.Dataset.zip((dataset_a, dataset_b)):
            step += 1
            # Perform one training step and get the losses
            gen_a2b_loss, gen_b2a_loss, disc_a_loss, disc_b_loss = train_step(real_a, real_b, generator_a2b, generator_b2a, discriminator_a, discriminator_b, gen_a2b_optimizer, gen_b2a_optimizer, disc_a_optimizer, disc_b_optimizer)

            checkpoint.step.assign(step)

            # Print training progress every 10 steps
            if step % 10 == 0:
                print(f"Epoch: {checkpoint.epoch.numpy()}, Step: {step}")
                print(f"Generator A2B Loss: {gen_a2b_loss:.4f}, Generator B2A Loss: {gen_b2a_loss:.4f}")
                print(f"Discriminator A Loss: {disc_a_loss:.4f}, Discriminator B Loss: {disc_b_loss:.4f}")

            # Break the loop when the desired number of steps per epoch is reached
            if step % steps_per_epoch == 0:
                break

        val_accuracy = calculate_validation_accuracy(val_dataset_a, val_dataset_b, generator_a2b, generator_b2a)
        val_accuracies.append(float(val_accuracy))
        print(f"Validation Mean Squared Error: {val_accuracy:.4f}")

        checkpoint.epoch.assign_add(1)

        # Save checkpoint after each epoch
        checkpoint_manager.save()
        print(f"Saved checkpoint at step {step}")

        # Save validation accuracies to a file
        with open(val_accuracies_path, "w") as f:
            json.dump(val_accuracies, f)


def calculate_validation_accuracy(val_dataset_a, val_dataset_b, generator_a2b, generator_b2a):
    mse = tf.keras.losses.MeanSquaredError()

    mse_values = []
    for real_a, real_b in tf.data.Dataset.zip((val_dataset_a, val_dataset_b)):
        fake_b = generator_a2b(real_a, training=False)
        fake_a = generator_b2a(real_b, training=False)
        
        recon_a = generator_b2a(fake_b, training=False)
        recon_b = generator_a2b(fake_a, training=False)
        
        mse_a = mse(real_a, recon_a)
        mse_b = mse(real_b, recon_b)

        mse_values.append((mse_a + mse_b) / 2.0)

    return np.mean(mse_values)



## Training Script ##



# Load train datasets
'''
apples_dataset = load_dataset("apple2oranges/apples/train", batch_size=1)
oranges_dataset = load_dataset("apple2oranges/oranges/train", batch_size=1)
apples_val_dataset = load_dataset("apple2oranges/apples/val", batch_size=1)
oranges_val_dataset = load_dataset("apple2oranges/oranges/val", batch_size=1)
'''

apples_train_dataset = load_dataset("apple2oranges/apples/train", batch_size=4)
oranges_train_dataset = load_dataset("apple2oranges/oranges/train", batch_size=4)
apples_val_dataset = load_dataset("apple2oranges/apples/val", batch_size=4)
oranges_val_dataset = load_dataset("apple2oranges/oranges/val", batch_size=4)


# Get checkpoint directory
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# This value gets adjusted according to the size of the datasets
steps_per_epoch = 30

# Train the model using the latest saved generator models
train(apples_train_dataset, oranges_train_dataset, apples_val_dataset, oranges_val_dataset, epochs=1000, steps_per_epoch=steps_per_epoch, checkpoint_dir=checkpoint_dir)
