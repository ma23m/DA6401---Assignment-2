# data_loader.py
import torch  # import PyTorch for tensor operations and device handling
from torchvision import transforms  # import image transformation utilities
from torchvision.datasets import ImageFolder  # import dataset wrapper for folder structure
from torch.utils.data import DataLoader, random_split  # import data loader and dataset splitting utility

# Define transformations for training images
train_transform = transforms.Compose([  # chain multiple transforms together
    transforms.Resize((224, 224)),  # resize images to 224x224
    transforms.RandomHorizontalFlip(),  # randomly flip images horizontally
    transforms.RandomRotation(15),  # randomly rotate images by up to 15 degrees
    transforms.ToTensor(),  # convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # normalize tensor channels with ImageNet stats
                         std=[0.229, 0.224, 0.225])  # use standard deviations for normalization
])

# Define transformations for test/validation images
test_transform = transforms.Compose([  # chain transforms for test data
    transforms.Resize((224, 224)),  # resize images to 224x224
    transforms.ToTensor(),  # convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # normalize with same stats as training
                         std=[0.229, 0.224, 0.225])  # use standard deviations for normalization
])

def get_dataloaders(train_path, val_path, batch_size=32, val_split=0.2, num_workers=2):  # function to create data loaders
    train_data = ImageFolder(root=train_path, transform=train_transform)  # load training images from folder
    test_data = ImageFolder(root=val_path, transform=test_transform)  # load validation images from folder

    train_size = int((1 - val_split) * len(train_data))  # compute size of training split
    val_size = len(train_data) - train_size  # compute size of validation split
    train_dataset, val_dataset = random_split(train_data, [train_size, val_size])  # split dataset

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)  # loader for training
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)  # loader for validation
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)  # loader for testing

    return train_loader, val_loader, test_loader, train_data  # return loaders and the original training dataset

# Choose computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use GPU if available, otherwise CPU

# Get data loaders using specified paths
train_path = r'C:\Users\HP\Downloads\nature_12K\train'  # path to training data
val_path = r'C:\Users\HP\Downloads\nature_12K\val'  # path to validation data
train_loader, val_loader, test_loader, train_data = get_dataloaders(train_path, val_path)  # initialize loaders

# # Initialize model (example usage commented out)
# model = CustomCNN(
#     input_channels=3,  # three-channel RGB images
#     input_size=128,  # input image size
#     num_classes=num_classes,  # number of output classes
#     num_conv_layers=5,  # number of convolutional layers
#     num_filters=[32, 64, 128, 256, 512],  # list of filter sizes
#     kernel_size=[3, 5, 3, 5, 1],  # list of kernel sizes
#     dense_neurons=[512, 256, 64]  # sizes of fully connected layers
# ).to(device)  # move model to chosen device