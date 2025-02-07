# Flower Image Classifier

## Overview

This project is an AI-powered image classifier that can identify different species of flowers. It was developed as part of the AWS AI & ML Scholarship program in collaboration with Udacity. The classifier uses deep learning techniques and transfer learning with EfficientNet architectures to achieve high accuracy in flower species identification.

## Features

- Train a neural network on a dataset of flower images
- Use transfer learning with EfficientNetV2-S or EfficientNetB0 architectures
- Predict flower species from new images
- Command-line interface for both training and prediction

## Requirements

- Python 3.x
- PyTorch
- torchvision
- NumPy
- Matplotlib
- Pillow

## Installation

1. Clone this repository:
git clone https://github.com/yourusername/ai-image-classifier.git
cd ai-image-classifier
text

2. Install the required packages:
pip install -r requirements.txt
text

## Usage

### Training

To train the model, use the `train.py` script:

python train.py data_directory [--save_dir SAVE_DIR] [--arch {efficientnet_v2_s,efficientnet_b0}]
[--learning_rate LEARNING_RATE] [--hidden_units HIDDEN_UNITS]
[--epochs EPOCHS] [--gpu]
text

**Arguments:**
- `data_directory`: Path to the folder of flower images
- `--save_dir`: Directory to save checkpoints (default: 'checkpoints')
- `--arch`: Model architecture to use (default: 'efficientnet_v2_s')
- `--learning_rate`: Learning rate for training (default: 0.001)
- `--hidden_units`: Number of hidden units in classifier (default: 512)
- `--epochs`: Number of training epochs (default: 5)
- `--gpu`: Use GPU for training if available

### Prediction

To predict the species of a flower from an image, use the `predict.py` script:

python predict.py image_path checkpoint [--top_k TOP_K] [--category_names CATEGORY_NAMES] [--gpu]
text

**Arguments:**
- `image_path`: Path to the image file
- `checkpoint`: Path to the saved model checkpoint
- `--top_k`: Return top K most likely classes (default: 5)
- `--category_names`: Path to JSON file mapping categories to real names
- `--gpu`: Use GPU for inference if available

## Project Structure

- `train.py`: Script for training the model
- `predict.py`: Script for making predictions on new images
- `cat_to_name.json`: JSON file mapping category labels to flower names
  - Note: this is automatically created after running `train.py`

## Acknowledgements

This project was completed as part of the AWS AI & ML Scholarship program in collaboration with Udacity. Special thanks to the mentors and instructors who provided guidance throughout the development process.
