import torch
import torchvision
import torchvision.transforms as transforms

from utils import readf

from tqdm import tqdm

Rootdir = readf('cache/rootdir')
# Import MNIST dataset

def get_dataset(name='cifar10', size=32):
    # channels = 1 if name == 'mnist' else 3 transforms.Normalize((0.5,)*channels, (0.5,)*channels)
    _transf = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    if name == 'mnist':
        trainset = torchvision.datasets.MNIST(root=Rootdir, train=True, download=True, transform=_transf)
        testset = torchvision.datasets.MNIST(root=Rootdir, train=False, download=True, transform=_transf)
    elif name == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=Rootdir, train=True, download=True, transform=_transf)
        testset = torchvision.datasets.CIFAR10(root=Rootdir, train=False, download=True, transform=_transf)
    elif name == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=Rootdir, train=True, download=True, transform=_transf)
        testset = torchvision.datasets.CIFAR100(root=Rootdir, train=False, download=True, transform=_transf)
    return trainset, testset

#image dataloader
def get_dataloader(trainset, testset=None, batch_size=64):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    if testset is None:
        return trainloader
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

def tokenize_dataset(_loader, iptmodel, device=0):
    iptmodel.to(device)
    tok_set = []
    for images, labels in tqdm(_loader, ncols=90, desc='tokenizing', unit='images', leave=False):
        images = images.to(device)
        patches = iptmodel.patcher(images)
        token_probability = iptmodel.tokenizer(patches)
        tok_images = torch.argmax(token_probability, dim=2)  # Assign the largest element in tok_image to tok_image
        tok_images = tok_images.to('cpu')
        
        tok_set.extend([(tok_images[i], labels[i].item()) for i in range(tok_images.size(0))])
    return tok_set


