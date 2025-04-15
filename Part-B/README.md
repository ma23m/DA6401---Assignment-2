# Fine-tuning ResNet50 for Image Classification on iNaturalist

### Model:
  
-  ResNet50 (pretrained)

### Dataset: 

 - iNaturalist 12K

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


### Validation:

After training, the model is evaluated on the validation set.

Validation accuracy is .

### Results:

The sweep runs  and the best performing hyperparameters are automatically selected based on validation accuracy.
