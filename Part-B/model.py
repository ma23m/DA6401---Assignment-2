import torch.nn as nn  # import neural network modules from PyTorch
from torchvision import models  # import pre-built vision models

def build_model(freeze_up_to):  # function to create and configure a ResNet50 model
    # load a pretrained ResNet50 model
    model = models.resnet50(pretrained=True)

    # get number of input features for the final fully connected layer
    num_ftrs = model.fc.in_features
    # replace the final layer so the model outputs 10 classes
    model.fc = nn.Linear(num_ftrs, 10)

    # freeze or unfreeze layers based on the freeze_up_to index
    child_counter = 0  # counter to track layer index
    for child in model.children():  # iterate through top-level layers
        if child_counter < freeze_up_to:  # if layer index is below threshold
            # disable gradient updates for all parameters in this layer
            for param in child.parameters():
                param.requires_grad = False
        else:  # if layer index is at or above threshold
            # enable gradient updates for all parameters in this layer
            for param in child.parameters():
                param.requires_grad = True
        child_counter += 1  # move to next layer index

    return model  # return the configured model
