import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader

from utils import Rootdir

# don't do normalization for more concistent range with norm-bounded attacks

def get_datasets(args):
    data_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.Grayscale(num_output_channels=3) if args.dataset == 'mnist' else (lambda x: x),
        transforms.ToTensor(),
    ])

    train_set = getattr(datasets, args.dataset.upper())(root=Rootdir, train=True, download=True, transform=data_transform)
    test_set = getattr(datasets, args.dataset.upper())(root=Rootdir, train=False, download=True, transform=data_transform)
    return train_set, test_set

def get_dataloader(args): 
    train_set, test_set = get_datasets(args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader


