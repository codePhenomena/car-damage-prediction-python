Car Damage Classification using ResNet50

Overview

This project is a deep learning-based car damage classification model using a fine-tuned ResNet50. The model classifies images into six categories:

F_Breakage (Front Breakage)

F_Crushed (Front Crushed)

F_Normal (Front Normal)

R_Breakage (Rear Breakage)

R_Crushed (Rear Crushed)

R_Normal (Rear Normal)

Model Details

The model utilizes a pre-trained ResNet50 with transfer learning. The earlier layers are frozen to retain feature extraction capabilities, while layer4 and the fully connected (fc) layer are fine-tuned for classification.

Installation

Ensure you have the necessary dependencies installed:

pip install -r requirements.txt

Usage

1. Load the Model and Make Predictions

import torch
from PIL import Image
from model import CarClassificationResNet, predict  # Ensure the script is correctly structured

image_path = "path_to_image.jpg"
result = predict(image_path)
print(f"Predicted Class: {result}")

2. Model Training (if needed)

If you wish to retrain or fine-tune the model, ensure you have a dataset and modify the CarClassificationResNet class accordingly.

Model Architecture

Uses torchvision.models.resnet50 with pre-trained weights

Freezes initial layers except layer4

Modifies the fc layer for six-class classification

File Structure

project/
│── model.py   # Contains CarClassificationResNet class and predict function
│── saved_model.pth  # Trained model weights
│── README.md  # Project documentation
│── test_image.jpg  # Example test image

Notes

Ensure saved_model.pth is present in the correct directory (model/)

Use correctly formatted images with RGB channels

Input images are resized to 224x224 pixels before feeding into the model

License

This project is open-source and can be modified for further improvements.