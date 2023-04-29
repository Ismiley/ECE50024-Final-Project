import os
import cv2
import numpy as np

def edge_detection(image_path):
    # Read the input image
    img = cv2.imread(image_path)

    # Check if the image was loaded correctly
    if img is None:
        print(f"Error: Could not load image '{image_path}'")
        return None

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Apply the Canny edge detection algorithm
    edges = cv2.Canny(blurred_img, 50, 150)

    return edges

def main():
    input_folder = "edge_detection_inputs"
    output_folder = "edge_detection_outputs"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if os.path.isfile(file_path):
            # Apply edge detection on the input image
            edges = edge_detection(file_path)

            if edges is not None:
                # Save the output image in the output folder
                output_path = os.path.join(output_folder, file_name)
                cv2.imwrite(output_path, edges)

if __name__ == "__main__":
    main()
