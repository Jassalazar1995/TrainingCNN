# MNIST Digit Classifier using Convolutional Neural Network (CNN)

## Overview
This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset. The network is trained on a set of labeled images and then evaluated for accuracy on a separate test set.

## Requirements
- Python 3.x
- PyTorch
- torchvision

## Network Architecture
The CNN (`Net` class) includes the following layers:
- Two convolutional layers (`conv1` and `conv2`).
- Two max pooling layers.
- Two fully connected layers (`fc1` and `fc2`).

## Dataset
The MNIST dataset, consisting of grayscale images of handwritten digits (0-9), is used. The dataset is automatically downloaded using `torchvision`.

## Training
The network is trained using the following settings:
- Loss function: Cross-Entropy Loss.
- Optimizer: Adam.
- Batch size: 64.

## Evaluation
After training, the model is evaluated on the test dataset to calculate its accuracy in digit classification.

## Usage
To train and evaluate the model, run the provided Python script. The script includes data loading, model initialization, training, and evaluation phases.

