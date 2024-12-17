import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, random_split

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

    train_size = int(len(train_set) * (1 - args.percent_val_split/100))
    val_size = len(train_set) - train_size
    train_set, val_set = random_split(train_set, [train_size, val_size])
    return train_set, val_set, test_set

def get_dataloader(args): 
    train_set, val_set, test_set = get_datasets(args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


