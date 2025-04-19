# train.py
import torch  # import PyTorch core library
import torch.nn as nn  # import neural network modules
import wandb  # import Weights & Biases for experiment tracking
from model import CustomCNN  # import the custom CNN model
from data_loader import train_loader, val_loader, device  # import data loaders and device setting

# Evaluation function to compute validation accuracy and loss
def evaluate_model(model, data_loader, criterion):
    model.eval()  # set model to evaluation mode
    correct, total, running_loss = 0, 0, 0.0  # initialize counters

    with torch.no_grad():  # disable gradient calculation
        for images, labels in data_loader:  # loop over validation batches
            images, labels = images.to(device), labels.to(device)  # move data to GPU/CPU
            outputs = model(images)  # forward pass
            loss = criterion(outputs, labels)  # compute batch loss
            running_loss += loss.item()  # accumulate loss

            _, predicted = torch.max(outputs, 1)  # get predicted classes
            total += labels.size(0)  # count total samples
            correct += (predicted == labels).sum().item()  # count correct predictions

    accuracy = 100 * correct / total  # calculate accuracy percentage
    avg_loss = running_loss / len(data_loader)  # calculate average loss per batch
    return accuracy, avg_loss  # return metrics

# Training function
def train_model(config=None):
    with wandb.init(config=config):  # initialize W&B run
        config = wandb.config  # get configuration settings

        # create model instance with config parameters
        model = CustomCNN(
            input_channels=3,
            num_filters=config.num_filters,
            activation=config.activation,
            dense_neurons=config.dense_neurons,
            num_classes=10,
            dropout_rate=config.dropout_rate,
            batch_norm=config.batch_norm
        ).to(device)  # move model to GPU/CPU

        criterion = nn.CrossEntropyLoss()  # define loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # set optimizer and learning rate

        for epoch in range(config.epochs):  # loop over epochs
            model.train()  # set model to training mode
            running_loss, correct, total = 0.0, 0, 0  # reset counters

            for images, labels in train_loader:  # loop over training batches
                images, labels = images.to(device), labels.to(device)  # move batch to device
                optimizer.zero_grad()  # clear gradients
                outputs = model(images)  # forward pass
                loss = criterion(outputs, labels)  # compute loss
                loss.backward()  # backward pass
                optimizer.step()  # update weights

                running_loss += loss.item()  # accumulate training loss
                _, predicted = torch.max(outputs, 1)  # get predictions
                total += labels.size(0)  # count total samples
                correct += (predicted == labels).sum().item()  # count correct predictions

            train_acc = 100 * correct / total  # calculate training accuracy
            val_acc, val_loss = evaluate_model(model, val_loader, criterion)  # evaluate on validation data

            print({  # print epoch summary
                'epoch': epoch + 1,
                'train_loss': running_loss / len(train_loader),
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            })

            wandb.log({  # log metrics to W&B
                'epoch': epoch + 1,
                'train_loss': running_loss / len(train_loader),
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            })
