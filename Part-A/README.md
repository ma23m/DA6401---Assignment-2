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

The dataset used for this project is the [inaturlist_12K](/kaggle/input/inaturalist-12).This dataset is a collection of labeled images where each image belongs to one of several classes. The model is trained to classify these images correctly. In the code, the dataset is accessed through ```train_loader```,``` val_loader```, and ```test_loader```, which represent the training, validation, and test sets, respectively. The images are fed into the model, and the goal is for the model to predict the correct class for each image.

## Data Processing:

 In this project, I used standard data loading and preprocessing steps to prepare the image dataset for training, validation, and testing.I applied transformations to the images like resizing, normalizing, and  data augmentation. These steps help the model generalize better.

## Model Architecture:

- The model is defined in the CustomCNN class ``` model = CustomCNN(...))```

- The CNN takes:

     1.```input_channels```:input channels = 3 (for RGB images)

     2.```num_filters```: num_filters is a list of filters per conv layer, (e.g. [64, 64, 64, 64, 64])

     3.```activation ```: The activation function are (e.g., 'relu', 'gelu','mish' etc.)
  
     4.```batch_norm```:batch_norm to improve training stability.
  
     5.```dropout_rate```:dropout_rate to avoid overfitting.
  
     6.```kernel_size```:A list specifying the kernel size for each convolutional layer.
  
     7.```dense_neurons```:The number of neurons in the dense layer.
  
     8.```Sweep method```: bayes (Bayesian optimization)

The example of the initializing model:

```model = CustomCNN(input_channels=3,input_size=128,num_classes=num_classes,num_conv_layers=5,num_filters=[32, 64, 128, 256, 512,1024],kernel_size=[3, 5, 3, 5, 1],dense_neurons=[512, 256, 64])```


## Training Process:

  The CNN model trained with selected configuration where 
  
  - ```Training loop logs```: train_loss, train_accuracy, val_loss, val_accuracy

  - ```Optimizer```: Adam which updated model parameters

  - ```Loss```: CrossEntropyLoss loss function used this model.
    

## Best Model Selection:

 - Best config manually inserted and model re-trained.

 - Evaluated on test set to report test_accuracy

## Prediction Visualization:

 - Displays 30 test samples with predicted and true labels.

## Customization:

Anyone can easily change the model by:

 - Updating the parameters in the sweep configuration (like number of filters, activation function, dropout, etc.).

 - Editing the ```CustomCNN()``` class to modify the CNN architecture as per need.

This lets anyone experiment with different model designs and find the one that works best for that dataset.


## Sweep Configuration

The sweep uses *Bayesian Optimization* to find the best hyperparameters that maximize validation accuracy. The table below summarizes the hyperparameters being explored:

| Hyperparameter       | Values/Options                                                                 |
|----------------------|--------------------------------------------------------------------------------|
| Convolution Filters  | [16, 32, 64, 128, 256], [64, 64, 64, 64, 64], [256, 128, 64, 32, 16], etc. |
| Activation Functions | relu, gelu, silu, mish                                                 |
| Batch Normalization  | True, False                                                                |
| Dropout Rates        | 0.2, 0.3                                                                   |
| Data Augmentation    | True, False                                                                |
| Filter Organization  | same, double, half                                                       |
| Kernel Sizes         | [3, 3, 3, 3, 3], [3, 3, 5, 3, 3], [3, 5, 3, 5, 3], [5, 5, 5, 5, 5], etc. |
| Dense Neurons        | [512], [256], [64, 128, 256], [512, 256, 64], etc.                      |
| Epochs               | 5, 7, 10, 12, 15, 17, 20                                         |





## Command-Line Arguments

The script supports the following arguments to customize the sweep:

| Argument          | Type   | Default                       | Description                          |
|-------------------|--------|-------------------------------|--------------------------------------|
| --project       | str  | 'DL_A2'                     | WandB project name                   |
| --sweep_name    | str  | 'scratch_hyperparam_sweep-1'| Name assigned to the sweep          |
| --sweep_count   | int  | 100                         | Number of sweep runs to execute     |


## Running the Code
To run a W&B sweep, use the following command:
```
cd <current_directory>
python wandb_login.py
python sweep_runner.py --project <project_name> --sweep_name <sweep_name> --sweep_count <number_of_runs>
```

### Example
```
cd <current_directory>
python wandb_login.py
python sweep_runner.py --project DL_A2 --sweep_name scratch_hyperparam_sweep-1 --sweep_count 1
```

### For the executing the best model please run the following command:
```
python best_model_run.py --num_filters 64 64 64 64 64 --kernel_size 3 5 3 5 3 --activation gelu --dropout_rate 0.2 --batch_norm --dense_neurons 128 --epochs 10 --run_name "best-run"
```

### For getting the 10×3 grid containing sample images from the test data and predictions made by my best model to wandb log please run:
```
python show_predictions.py
```

## Note:

The model is trained using a  ```CUDA-enabled GPU``` (available on ```Kaggle```), since training CNNs requires a lot of computation and would be slow on a regular ```CPU```.
