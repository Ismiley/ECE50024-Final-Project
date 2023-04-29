import json
import matplotlib.pyplot as plt

def plot_validation_accuracies(json_file, output_image):
    # Load validation accuracies from the JSON file
    with open(json_file, "r") as f:
        val_accuracies = json.load(f)

    # Generate a list of epoch numbers
    epochs = list(range(1, len(val_accuracies) + 1))

    # Plot the validation accuracies against the epochs
    plt.plot(epochs, val_accuracies, marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Mean Squared Error")
    plt.title("Validation Mean Squared Error vs Epochs")
    plt.grid()

    # Save the plot as an image
    plt.savefig(output_image)
    plt.show()

# Specify the JSON file containing the validation accuracies
json_file = "checkpoints/val_accuracies.json"

# Specify the output image filename
output_image = "validation_accuracy_plot.png"

# Plot the validation accuracies
plot_validation_accuracies(json_file, output_image)
