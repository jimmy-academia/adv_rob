import torch
import torchvision
import torchvision.transforms as transforms

Rootdir = '/home/jimmyyeh/Documents/CRATER/DATASET'
# Import MNIST dataset

def get_dataset(name='mnist'):
    if name == 'mnist':
        trainset = torchvision.datasets.MNIST(root=Rootdir, train=True, download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.MNIST(root=Rootdir, train=False, download=True, transform=transforms.ToTensor())
    elif name == 'cifar10':
    # Import CIFAR-10 dataset
        trainset = torchvision.datasets.CIFAR10(root=Rootdir, train=True, download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.CIFAR10(root=Rootdir, train=False, download=True, transform=transforms.ToTensor())
    return trainset, testset

#image dataloader
def get_dataloader(trainset, testset, batch_size=64):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader