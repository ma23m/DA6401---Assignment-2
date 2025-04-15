# ```PART: A```
# Image Classification with Convolutional Neural Network(CNNs)


## Introduction:

In this project, I built and trained a Convolutional Neural Network (CNN) model from scratch using PyTorch.This model  classify images into different categories.I used the Weights & Biases (WandB) tool to perform hyperparameter tuning through a sweep. This sweep helped me to find the best model configuration. The model was trained on a dataset using various settings like different numbers of filters, activation functions, dropout rates and more. After training, i evaluated the model's performance on the validation and test sets, and visualized some sample predictions to understand how well the model performed.

## Core Libraries:

```
torch

torchvision

numpy

matplotlib

```
## Dataset:

The dataset used for this project is the [inaturlist_12K](https://drive.google.com/drive/folders/17JUjrqNAopZXWyB7xmuew1WvIxKXjpd5?usp=sharing).This dataset is a collection of labeled images where each image belongs to one of several classes. The model is trained to classify these images correctly. In the code, the dataset is accessed through ```train_loader```,``` val_loader```, and ```test_loader```, which represent the training, validation, and test sets, respectively. The images are fed into the model, and the goal is for the model to predict the correct class for each image.

## Data Processing:

 In this project, I used standard data loading and preprocessing steps to prepare the image dataset for training, validation, and testing.I applied transformations to the images like resizing, normalizing, and  data augmentation. These steps help the model generalize better.

## Model Architecture:

- The model is defined in the CustomCNN class ``` model = CustomCNN(...))```

- The CNN takes:

     1.```input_channels```:input channels = 3 (for RGB images)

     2.```num_filters```: num_filters is a list of filters per conv layer, (e.g. [64, 64, 64, 64, 64])

     3.```activation ```: The activation function are (e.g., 'relu', 'gelu','mish' etc.)

dense_neurons = 128 (for the fully connected layer)

dropout_rate to avoid overfitting

batch_norm to improve training stability







