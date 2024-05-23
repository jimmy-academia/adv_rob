import torch
import torchvision
from torchvision import transforms

from pathlib import Path
import warnings

def get_dataloader(dataset='mnist', batch_size=128, dataset_root='../../DATASET/'):
    """
    Loads the specified dataset using torchvision and returns the dataloaders for training and testing.

    Args:
        dataset (str): Name of the dataset to load. Defaults to 'mnist'.
        batch_size (int): Batch size for the dataloaders. Defaults to 128.
        dataset_root (str): Root directory for downloading datasets. Defaults to '../DATASET/'.

    Returns:
        tuple: A tuple containing the training and testing dataloaders.
    """
    dataset_path = Path(dataset_root)
    assert dataset_path.exists(), warnings.warn(f"Dataset root path '{dataset_path.absolute()}' does not exist.", RuntimeWarning)

    # Lists of grayscale and RGB datasets
    grayscale_datasets = ['mnist']
    rgb_datasets = ['cifar10', 'imagenet', 'svhn', 'cifar100']  # Extend this list with more RGB datasets

    # Determine normalization based on dataset type
    if dataset in grayscale_datasets:
        normalization = transforms.Normalize((0.5,), (0.5,))
    elif dataset in rgb_datasets:
        normalization = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    else:
        raise ValueError(f"{dataset} dataset not supported.")

    # Create the transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalization
    ])

    # Dynamically load the dataset class
    dataset_class = None
    for name, cls in torchvision.datasets.__dict__.items():
        if isinstance(cls, type) and name.lower() == dataset.lower():
            dataset_class = cls
            break

    if not dataset_class:
        raise ValueError(f"{dataset} dataset class not found in torchvision.datasets.")

    traindata = dataset_class(dataset_root, train=True, download=True, transform=transform)
    testdata = dataset_class(dataset_root, train=False, download=True, transform=transform)

    # Create dataloaders
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return trainloader, testloader
