Brain Tumor Classification using VGG16 Transfer Learning 🧠

📋 Overview

This repository contains the code for a deep learning model that classifies MRI images of brain tumors into four categories: glioma, meningioma, pituitary tumor, or no tumor (notumor).

The model utilizes Transfer Learning by employing the pre-trained VGG16 convolutional neural network as a feature extractor. The VGG16 base layers are frozen to leverage the weights learned from the vast ImageNet dataset, and a custom classification head is trained on top of its output, significantly speeding up training and improving performance on a limited medical image dataset.

✨ Features

    Transfer Learning: Uses the VGG16 architecture with ImageNet weights.

    Frozen Base: The VGG16 base is frozen to act as a stable feature extractor.

    Custom Classification Head: A Flatten, Dense (512, ReLU), Dropout (0.5), and Dense (4, Softmax) sequence is used for the final classification.

    Data Augmentation: Extensive augmentation is applied to the training data to improve generalization.

    Inference Script: A separate script is provided to load the trained model and predict the class of a new single image, complete with visual output.

🛠️ Requirements

Software

    Python 3.x

Python Libraries

The core libraries required are:
Bash

tensorflow
numpy
matplotlib
Pillow # Used implicitly by tensorflow.keras.preprocessing.image

You can install them using pip:
Bash

pip install tensorflow numpy matplotlib

📂 Project Structure

This project assumes the following file and directory structure for data and code:

brain_tumor_classification/
├── brain_tumor_vgg16_transfer_model.h5 # Saved model (output after training)
├── brain_tumor_training.py            # Code for model definition, training, and saving
├── brain_tumor_prediction.py          # Code for loading the model and making single-image predictions
├── data/
│   ├── Training/                      # Directory for training and validation data
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/                       # Directory for test data
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
└── README.md

Note: The provided code snippets in your input use paths like /content/drive/MyDrive/brain_tumor_detection/Training. You must adjust these paths in the training script to point to your actual data directories.

🚀 Usage

1. Training the Model

The training script (brain_tumor_training.py) defines the VGG16 model with the custom head, compiles it, and trains it using the defined data generators.

    Ensure Data Paths are Correct: Open brain_tumor_training.py and verify the directory paths in the flow_from_directory calls match your setup.

    Run the Training Script:
    Bash

    python brain_tumor_training.py

    Output: The script will train for 25 epochs (default), print the model summary, display the final test accuracy, and save the model as brain_tumor_vgg16_transfer_model.h5.

2. Making Predictions

The prediction script (brain_tumor_prediction.py) handles loading the trained model and classifying a new, single MRI scan.

    Adjust Model Path: Ensure the MODEL_PATH variable in brain_tumor_prediction.py correctly points to your saved .h5 file.
    Python

MODEL_PATH = 'brain_tumor_vgg16_transfer_model.h5'

Specify New Image Path: Crucially, update the NEW_IMAGE_PATH variable to the absolute or relative path of the new MRI image you wish to classify.
Python

NEW_IMAGE_PATH = '/path/to/your/new_mri_scan.jpg'

Run the Prediction Script:
Bash

    python brain_tumor_prediction.py

    Output: The script will display the input image, print the predicted class (e.g., meningioma), and the confidence score.

⚙️ Model Details

This table summarizes the architecture and training settings used for the image classification model.
<img width="923" height="589" alt="image" src="https://github.com/user-attachments/assets/0f9030a1-016e-42c5-b260-46658a8ab632" />
