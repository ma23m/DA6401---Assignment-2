# Fine-tuning ResNet50 for Image Classification on iNaturalist

### Model:
  
-  I have chosen ```ResNet50``` for fine-tuning in Part B of the project.

### Dataset: 

 - Dataset used: [iNaturalist 12K](/kaggle/input/inaturalist-12)
   
 - The dataset is divided into training and validation sets.
   
 - Preprocessing includes:
   
   - Resizing images to 224x224 pixels.

   - Random cropping and horizontal flipping for data augmentation.

   - Normalization using ImageNet mean and standard deviation.
   

### Model Configuration:

  - A ResNet50 model with pretrained weights (trained on ImageNet) is used.

  - The final fully connected (FC) layer is replaced with:

             nn.Linear(num_ftrs, 10)
   
      to adapt to the 10-class classification task.

  - Only the weights of the final FC layer and optionally some later layers are trained; early layers can be frozen based on the ```freeze_up_to``` hyperparameter.

### Fine-Tuning Technique:

  - Strategy:
   - Partial fine-tuning (selective layer unfreezing)

  - Layer Freezing:

    - Layers of ResNet50 are frozen up to a configurable depth (freeze_up_to).

    - This helps retain low-level feature representations while adapting high-level features to the new dataset.

  - Final Layer:
    
    - Always trainable and modified to output predictions for 10 classes.

### Hyperparameters Tuned:

  - ```Batch size```: [16, 32, 64, 128]

  - ```Number of epochs```: [5, 7, 10, 12]

  - ```Learning rate```: [0.1, 0.01, 0.001]

  - ```Optimization Method```: Adam Optimizer

  - ```Loss Function```: CrossEntropyLoss


### Data Loading:
  - The iNaturalist 12K dataset is used for training and validation.

  - The dataset is preprocessed with resizing, normalization, and transformation.
 
### Training:

The model is trained 
  - ```batch_size```: [16, 32, 64, 128]

  - ```epochs```: [5, 7, 10, 12]

  - ```learning_rate```: [0.1, 0.01, 0.001]


### Validation:

After training, the model is evaluated on the validation set.

Validation accuracy is **`78.5%`**.

## Sweep Configuration

The sweep searches over the following hyperparameters:

| Hyperparameter  | Values                 | Description                                     |
|-----------------|------------------------|-------------------------------------------------|
| batch_size    | 16, 32, 64, 128        | Number of samples per training batch           |
| freeze_up_to  | 0, 3, 5, 7             | Number of layers to freeze in transfer learning|
| epochs        | 5, 7, 10, 12           | Total training epochs                          |
| learning_rate | 0.1, 0.01, 0.001       | Learning rate for the optimizer                |



## Command-Line Arguments

The script accepts the following arguments to configure the sweep:

| Argument           | Type    | Default                                         | Description                              |
|--------------------|---------|-------------------------------------------------|------------------------------------------|
| --project_name   | str   | 'DL_A2'                                       | W&B project name                         |
| --sweep_name     | str   | 'resnet50_hyperparam_sweep'                  | W&B sweep name                           |
| --data_path      | str   | '/kaggle/input/inaturalist-12/inaturalist_12K'| Root path to dataset                     |
| --sweep_count    | int   | 50                                            | Number of sweep runs to execute          |
| --use_cuda       | flag  | False (enabled when flag is present)         | Use CUDA for training if available       |


## Running the Code
To run a W&B sweep, please run the following command:
```
python train.py --project_name <project_name> --sweep_name <sweep_name> --data_path <path_to_image_dataset> --sweep_count 1 --use_cuda
```

## Example Usage
```
python train.py --project_name "DL_A2" --sweep_name "resnet50_exp2_sweep" --data_path "C:\Users\HP\Downloads\nature_12K" --sweep_count 1 --use_cuda
```


### Results:

The sweep runs  and the best performing hyperparameters are automatically selected based on validation accuracy.
