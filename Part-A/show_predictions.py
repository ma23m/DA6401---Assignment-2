import torch  # import PyTorch for tensor operations
import numpy as np  # import NumPy for array operations
import matplotlib.pyplot as plt  # import Matplotlib for plotting images
import wandb  # import Weights & Biases for logging results
from model import CustomCNN  # import the same model architecture used in training
from data_loader import test_loader, train_data, device  # import test data loader, dataset info, and device

# Utility to reverse normalization and convert tensor to image array
def imshow(img):
    img = img.cpu().numpy().transpose((1, 2, 0))  # move tensor to CPU and rearrange dimensions
    mean = np.array([0.485, 0.456, 0.406])  # ImageNet mean values
    std = np.array([0.229, 0.224, 0.225])  # ImageNet standard deviation values
    img = std * img + mean  # undo normalization
    img = np.clip(img, 0, 1)  # clamp values between 0 and 1
    return img  # return the processed image

# Function to display and log sample predictions
def show_predictions(model, dataloader, class_names, num_images=30):
    model.eval()  # set model to evaluation mode
    images_shown = 0  # counter for images displayed
    plt.figure(figsize=(12, 20))  # create a new figure with specified size

    with torch.no_grad():  # disable gradient tracking
        for inputs, labels in dataloader:  # iterate over test dataset
            inputs, labels = inputs.to(device), labels.to(device)  # move batch to device
            outputs = model(inputs)  # get model outputs
            _, preds = torch.max(outputs, 1)  # select class with highest score

            for j in range(inputs.size(0)):  # iterate over images in the batch
                if images_shown >= num_images:  # stop if reached limit
                    break  # exit inner loop
                img = imshow(inputs[j])  # unnormalize and convert image
                plt.subplot(10, 3, images_shown + 1)  # set subplot position
                plt.imshow(img)  # display the image
                plt.title(f"Pred: {class_names[preds[j]]} | True: {class_names[labels[j]]}", fontsize=8)  # set title
                plt.axis('off')  # turn off axis labels
                images_shown += 1  # update counter
            if images_shown >= num_images:  # check after batch
                break  # exit outer loop

    plt.tight_layout()  # adjust subplot layout
    plt.savefig("predictions.png", bbox_inches='tight')  # save figure to file
    plt.close()  # close the figure
    wandb.log({"sample_predictions": wandb.Image("predictions.png")})  # log image to W&B
    print("Prediction image saved to predictions.png and logged to W&B.")  # notify user

# Main function to load model and call prediction display
def main():
    wandb.init(project="DL_A2", name="show-predictions")  # start a new W&B run

    # recreate model architecture matching training setup
    model = CustomCNN(
        input_channels=3,  # RGB images
        num_filters=[64, 64, 64, 64, 64],  # convolutional filter sizes
        kernel_size=[3, 5, 3, 5, 3],  # convolutional kernel sizes
        activation='gelu',  # activation function
        dense_neurons=128,  # size of dense layer
        num_classes=len(train_data.classes),  # number of output classes
        dropout_rate=0.2,  # dropout rate
        batch_norm=True,  # enable batch normalization
        num_conv_layers=5  # number of conv layers
    ).to(device)  # move model to GPU or CPU

    model.load_state_dict(torch.load("best_model.pth", map_location=device))  # load trained weights
    print("Trained model loaded successfully.")  # notify user

    show_predictions(model, test_loader, train_data.classes, num_images=30)  # display and log predictions

    wandb.finish()  # end the W&B run

# Execute main function when script is run directly
if __name__ == "__main__":
    main()  # call main()
