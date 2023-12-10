
# ECE50024 Final Project: CycleGAN for Image-to-Image Translation

## Project Overview
This project implements a CycleGAN model for image-to-image translation, focusing on converting images of apples to oranges and vice versa. It utilizes deep learning techniques to learn the characteristics of each fruit and applies these learned features to transform images from one domain to the other.

## Features
- Image-to-image translation using CycleGAN.
- Training and testing capabilities for the model.
- Dataset handling for apples and oranges images.

## Prerequisites
- Python 3.9 or higher.
- PyTorch and other necessary Python libraries (see `requirements.txt`).

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/Ismiley/ECE50024-Final-Project.git
   ```
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Training the Model
Run the following command to train the CycleGAN model:
```
python3.9 train.py
```
Training involves learning transformations between apple and orange images.

## Testing the Model
To test the model, use:
```
python3.9 test.py
```
This will apply the learned transformations to test images and generate the output.

## Fetching the Dataset
- Download the dataset from [Berkeley EECS](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/).
- Organize the dataset into training and validation sets.

## Dataset Structure
Organize the dataset as follows:
```
apple2oranges/
--apples/
----train/
----val/
--oranges/
----train/
----val/
```

## File Descriptions
- `train.py`: Script for training the CycleGAN model.
- `test.py`: Script for testing the trained model.
- `cyclegan.py`, `dataset_loader.py`, `edgeDetection.py`, `helperFunctions.py`, `model.py`, `save_validation.py`: Supporting scripts for the model and data handling.

## Results and Evaluation
- The model's performance can be evaluated based on the quality of the image translation.
- `test_results_paper/`: Directory containing output results for comparison.
