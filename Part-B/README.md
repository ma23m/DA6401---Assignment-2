# Fine-tuning ResNet50 for Image Classification on iNaturalist

### Model:
  
-  I have chosen ```ResNet50``` for fine-tuning in Part B.

### Dataset: 

 - [iNaturalist 12K](/kaggle/input/inaturalist-12)

### Hyperparameters Tuned:

  - ```Batch size```: [16, 32, 64, 128]

  - ```Number of epochs```: [5, 7, 10, 12]

  - ```Learning rate```: [0.1, 0.01, 0.001]

  - ```Optimization Method```: Adam Optimizer

  - ```Loss Function```: CrossEntropyLoss

### Model Configuration:

  - A ResNet50 model is used with the pretrained weights.

  - The final fully connected layer is modified to predict 10 classes.

  - Only the weights of the final fully connected layer are trained; the rest are frozen.

### Data Loading:
  - The iNaturalist 12K dataset is used for training and validation.

  - The dataset is preprocessed with resizing, normalization, and transformation.
 
### Training:

The model is trained 
  - ```batch_size```: [16, 32, 64, 128]

  - ```epochs```: [5, 7, 10, 12]

  - ```learning_rate```: [0.1, 0.01, 0.001]

### Fine-Tuning Technique:
  
  - ```Fine-Tuning Strategy```: Partial fine-tuning (also known as selective layer unfreezing).

  - ```Layer Freezing```:
    - Initial layers of ResNet50 are optionally frozen based on the ```freeze_up_to``` hyperparameter.

    - This allows tuning of only the later layers while keeping early layers fixed.

  - ```Final Layer Modification```:

    - The original fully connected layer is replaced with a new one: nn.Linear(num_ftrs, 10) to adapt to the 10-class 
         classification task.

### Validation:

After training, the model is evaluated on the validation set.

Validation accuracy is .

### Results:

The sweep runs  and the best performing hyperparameters are automatically selected based on validation accuracy.
