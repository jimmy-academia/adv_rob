import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from utils import *

def get_dataloader(args):
    data_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])
    train_set = getattr(datasets, args.dataset.upper())(root=Rootdir, train=True, download=True, transform=data_transform)
    test_set = getattr(datasets, args.dataset.upper())(root=Rootdir, train=False, download=True, transform=data_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader

