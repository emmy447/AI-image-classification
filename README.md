# AI-image-classification
This project focuses on the following tasks:

Dataset Preprocessing:

Using a subset of the CIFAR-10 dataset (500 training and 100 test images per class).
Feature extraction using a pre-trained ResNet-18 model.
Dimensionality reduction with Principal Component Analysis (PCA).

AI Models:

Naive Bayes: Implemented from scratch and using Scikit-learn.
Decision Trees: Built from scratch with Gini coefficients and explored using Scikit-learn.
Multi-Layer Perceptron (MLP): Designed and trained using PyTorch.
Convolutional Neural Networks (CNN): Implemented a custom VGG11 network using PyTorch.

Evaluation Metrics:

Accuracy, Precision, Recall, and F1-Score. It also generates confusion matrices for each of the models
Comparison across models and their variants, exploring performance trade-offs.

Key Features

Feature Engineering:
Utilized pre-trained ResNet-18 for feature extraction.
Reduced feature dimensionality to 50 using PCA for efficient model training.

Model Experiments:
Tested varying depths for decision trees, MLP, and CNNs.
Explored different hidden layer sizes and kernel sizes in MLPs and CNNs.

Visualization:
Confusion matrices for analyzing classification performance.
Comprehensive metrics table summarizing findings across all models.

Technologies Used

Python Libraries:
PyTorch: For MLP and CNN implementation and training.
Scikit-learn: For Naive Bayes, decision trees, and PCA.
NumPy: For data manipulation and computations.

Dataset:
CIFAR-10, consisting of 32x32 RGB images from ten object classes.

Deliverables
Python Scripts: Code for data preprocessing, model training, and evaluation.
Trained Models: Saved weights for all AI models and their variants.
Evaluation Results: Metrics and insights from experiments documented in a detailed report.
