from torchvision import transforms  # import image transformation utilities
from torchvision.datasets import ImageFolder  # import dataset loader for folder structure
from torch.utils.data import DataLoader  # import DataLoader for batching data

def get_data_loaders(batch_size, data_path):  # function to create train and validation loaders
    from os.path import join  # import join to build file paths

    # define image transformations
    transform = transforms.Compose([  # chain multiple transforms
        transforms.Resize((224, 224)),  # resize images to 224x224
        transforms.RandomResizedCrop(224),  # crop and resize randomly to 224x224
        transforms.RandomHorizontalFlip(),  # randomly flip images horizontally
        transforms.ToTensor(),  # convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # normalize with ImageNet mean
                             std=[0.229, 0.224, 0.225])  # normalize with ImageNet std
    ])

    # load datasets from disk with transformations applied
    train_data = ImageFolder(join(data_path, 'train'), transform=transform)  # training dataset
    val_data = ImageFolder(join(data_path, 'val'), transform=transform)  # validation dataset

    # create data loaders for batching and shuffling
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)  # training loader with shuffling
    val_loader = DataLoader(val_data, batch_size=batch_size)  # validation loader without shuffling

    # return loaders and dataset sizes
    return {'train': train_loader, 'val': val_loader}, {'train': len(train_data), 'val': len(val_data)}  # return dicts of loaders and lengths