import torch  # PyTorch for tensor operations and model handling
import wandb  # Weights & Biases for experiment tracking
import torch.nn as nn  # Neural network module
import torch.optim as optim  # Optimizers like Adam, SGD, etc.
from model import build_model  # Custom function to create the model
from dataloader import get_data_loaders  # Function to load training and validation data

# Set device to GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training function
def train(args):
    with wandb.init() as run:  # Start a new wandb run
        config = run.config  # Load sweep configuration

        # Get training and validation dataloaders
        dataloaders, dataset_sizes = get_data_loaders(config.batch_size, args.data_path)

        # Build model and move to appropriate device
        model = build_model(config.freeze_up_to).to("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

        # Define loss function
        criterion = nn.CrossEntropyLoss()

        # Use Adam optimizer (only for the final FC layer)
        optimizer = optim.Adam(model.fc.parameters(), lr=config.learning_rate)

        # Learning rate scheduler to reduce LR after some epochs
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # Training loop
        for epoch in range(config.epochs):
            model.train()  # Set model to training mode
            running_loss = 0.0  # Track total loss
            running_corrects = 0  # Track total correct predictions

            # Iterate over training batches
            for inputs, labels in dataloaders['train']:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()  # Clear gradients
                outputs = model(inputs)  # Forward pass
                _, preds = torch.max(outputs, 1)  # Get predictions
                loss = criterion(outputs, labels)  # Compute loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

                # Accumulate loss and correct predictions
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Compute average loss and accuracy for the epoch
            epoch_loss = running_loss / dataset_sizes['train']
            epoch_acc = (running_corrects.double() / dataset_sizes['train']) * 100

            # Log metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "train_acc": epoch_acc
            })

            # Step the learning rate scheduler
            scheduler.step()

        # Validation after training
        model.eval()  # Set model to evaluation mode
        running_corrects = 0  # Track correct predictions

        # Iterate over validation data
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

        # Calculate and log validation accuracy
        val_acc = (running_corrects.double() / dataset_sizes['val']) * 100
        wandb.log({"val_acc": val_acc})
        print(f"[{run.name}] Validation Accuracy: {val_acc:.2f}%")
