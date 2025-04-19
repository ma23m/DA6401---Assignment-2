import torch  # import PyTorch for tensor operations and device control
import torch.nn as nn  # import neural network modules
import argparse  # import argparse for command-line argument parsing
import wandb  # import Weights & Biases for experiment tracking
from model import CustomCNN  # import the custom CNN model class
from data_loader import train_loader, val_loader, test_loader, train_data  # import data loaders and dataset info
from train import evaluate_model  # import evaluation helper function

# Set computation device: use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- ARGPARSE -------------------
def parse_args():  # function to parse CLI arguments
    parser = argparse.ArgumentParser(description="Train best model with custom config")  # create parser
    parser.add_argument('--num_filters', type=int, nargs='+', default=[64, 64, 64, 64, 64],  # list of conv filters
                        help="List of filters for each conv layer")
    parser.add_argument('--kernel_size', type=int, nargs='+', default=[3, 3, 3, 3, 3],  # list of kernel sizes
                        help="List of kernel_size for each conv layer")    
    parser.add_argument('--activation', type=str, default='gelu', choices=['relu', 'gelu', 'silu', 'mish'],  # activation choice
                        help="Activation function")
    parser.add_argument('--dropout_rate', type=float, default=0.2,  # dropout rate
                        help="Dropout rate")
    parser.add_argument('--batch_norm', action='store_true',  # flag to enable batch norm
                        help="Use batch normalization")
    parser.add_argument('--dense_neurons', type=int, default=128,  # dense layer neuron count
                        help="Number of neurons in dense layer")
    parser.add_argument('--epochs', type=int, default=20,  # number of epochs
                        help="Number of training epochs")
    parser.add_argument('--run_name', type=str, default='best-model-run',  # W&B run name
                        help="WandB run name")
    return parser.parse_args()  # return parsed arguments

# ------------------- MAIN FUNCTION -------------------
def main():  # main training and evaluation routine
    args = parse_args()  # parse command-line arguments

    wandb.init(  # initialize W&B run
        project="DL_A2",  # project name in W&B
        config=vars(args),  # log all args to W&B config
        name=args.run_name  # set run name
    )

    # instantiate the model with provided hyperparameters
    best_model = CustomCNN(
        input_channels=3,  # RGB images
        num_filters=args.num_filters,  # list of conv filter sizes
        kernel_size=args.kernel_size,  # list of kernel sizes
        activation=args.activation,  # activation function
        dense_neurons=args.dense_neurons,  # dense layer size
        num_classes=len(train_data.classes),  # number of classes from dataset
        dropout_rate=args.dropout_rate,  # dropout percentage
        batch_norm=args.batch_norm,  # whether to apply batch norm
        num_conv_layers=len(args.num_filters),  # number of conv layers
        input_size=224  # input image resolution
    ).to(device)  # move model to GPU/CPU

    criterion = nn.CrossEntropyLoss()  # define cross-entropy loss for classification
    optimizer = torch.optim.Adam(best_model.parameters(), lr=0.001)  # use Adam optimizer

    for epoch in range(args.epochs):  # loop over training epochs
        best_model.train()  # set model to training mode
        running_loss, correct, total = 0.0, 0, 0  # reset epoch metrics

        for images, labels in train_loader:  # iterate training batches
            images, labels = images.to(device), labels.to(device)  # move batch to device
            optimizer.zero_grad()  # clear previous gradients
            outputs = best_model(images)  # forward pass
            loss = criterion(outputs, labels)  # compute loss
            loss.backward()  # backpropagate
            optimizer.step()  # update weights

            running_loss += loss.item()  # accumulate batch loss
            _, predicted = torch.max(outputs, 1)  # get predicted classes
            total += labels.size(0)  # count total samples
            correct += (predicted == labels).sum().item()  # count correct predictions

        train_acc = 100 * correct / total  # compute training accuracy
        val_acc, val_loss = evaluate_model(best_model, val_loader, criterion)  # evaluate on validation set

        wandb.log({  # log epoch metrics to W&B
            "epoch": epoch + 1,  # epoch number
            "train_accuracy": train_acc,  # training accuracy
            "train_loss": running_loss / len(train_loader),  # training loss
            "val_accuracy": val_acc,  # validation accuracy
            "val_loss": val_loss  # validation loss
        })

        print(f"[Epoch {epoch+1}] Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")  # print epoch summary

    # ------------------- TESTING -------------------
    best_model.eval()  # set model to eval mode
    correct, total = 0, 0  # reset test metrics

    with torch.no_grad():  # disable gradient tracking for testing
        for images, labels in test_loader:  # iterate test batches
            images, labels = images.to(device), labels.to(device)  # move to device
            outputs = best_model(images)  # forward pass
            _, preds = torch.max(outputs, 1)  # get predictions
            correct += (preds == labels).sum().item()  # update correct count
            total += labels.size(0)  # update total count

    test_accuracy = correct / total  # compute test accuracy
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")  # print test accuracy
    wandb.log({"test_accuracy": test_accuracy * 100})  # log test accuracy to W&B

    # Save trained model parameters to file
    torch.save(best_model.state_dict(), "best_model.pth")  # save state dict
    print("Model saved to best_model.pth")  # confirm save

# ------------------- ENTRY POINT -------------------
if __name__ == "__main__":  # only run when executed directly
    main()  # call main function

# Example CLI:
# python best_model_run.py --num_filters 64 64 64 64 64 --kernel_size 3 5 3 5 3 --activation gelu \
#   --dropout_rate 0.2 --batch_norm --dense_neurons 128 --epochs 10 --run_name "custom-best-run"
