from model import *

def test(input_dataset, checkpoint_dir, output_dir):

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load pre-trained generator models
    checkpoint = Checkpoint(generator_a2b=build_generator((256, 256, 3)), generator_b2a=build_generator((256, 256, 3)))
    checkpoint_manager = CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    latest_checkpoint = checkpoint_manager.latest_checkpoint
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint).expect_partial()
        generator_a2b = checkpoint.generator_a2b
        generator_b2a = checkpoint.generator_b2a
    else:
        raise FileNotFoundError("No latest checkpoint found.")


    # Test the generators on the input dataset
    for i, image in enumerate(input_dataset):
        generated_b = generator_a2b(image, training=False)
        generated_a = generator_b2a(image, training=False)

        # Save the generated images
        save_image(generated_b[0].numpy(), os.path.join(output_dir, f'generated_b_{i}.jpg'))
        save_image(generated_a[0].numpy(), os.path.join(output_dir, f'generated_a_{i}.jpg'))

## Training Script ##

# Get checkpoint directory
checkpoint_dir = "checkpoints"

# Load test datasets
apples_test_dataset = load_dataset("testApples", batch_size=1)
oranges_test_dataset = load_dataset("testOranges", batch_size=1)

# Test the model using the latest saved generator models
test(apples_test_dataset, checkpoint_dir, "test_results_paper/apples")
test(oranges_test_dataset, checkpoint_dir, "test_results_paper/oranges")

