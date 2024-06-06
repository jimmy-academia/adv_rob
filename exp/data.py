import torch
import torchvision
import torchvision.transforms as transforms

from utils import readf, check

from tqdm import tqdm

Rootdir = readf('../cache/rootdir')
# Import MNIST dataset

def get_dataset(name='cifar10'):
    if name == 'mnist':
        mnist_transf = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.MNIST(root=Rootdir, train=True, download=True, transform=mnist_transf)
        testset = torchvision.datasets.MNIST(root=Rootdir, train=False, download=True, transform=mnist_transf)
    elif name == 'cifar10':
    # Import CIFAR-10 dataset
        trainset = torchvision.datasets.CIFAR10(root=Rootdir, train=True, download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.CIFAR10(root=Rootdir, train=False, download=True, transform=transforms.ToTensor())
    return trainset, testset

#image dataloader
def get_dataloader(trainset, testset=None, batch_size=64):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    if testset is None:
        return trainloader
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

# def split_patch(images, patch_size):
    # return images.view(images.size(0), -1, patch_size)
    
def tokenize_dataset(_loader, iptmodel, device=0):
    tok_set = []
    for images, labels in tqdm(_loader, ncols=90, desc='tokenizing', unit='images', leave=False):
        images = images.to(device)
        patches = iptmodel.patcher(images)
        token_probability = iptmodel.tokenizer(patches)
        tok_images = torch.argmax(token_probability, dim=2)  # Assign the largest element in tok_image to tok_image
        tok_images = tok_images.to('cpu')
        
        tok_set.extend([(tok_images[i], labels[i].item()) for i in range(tok_images.size(0))])
    return tok_set


def patchwise_loader(_loader, patch_size, batch_size=12800):
    all_patches = []
    for images, labels in tqdm(_loader, ncols=90, desc='patch_splitting', unit='images', leave=False):
        images = images.view(-1, patch_size)
        all_patches.append(images)

    patch_dataset = torch.utils.data.TensorDataset(torch.cat(all_patches))
    patch_loader = get_dataloader(patch_dataset, batch_size=batch_size)

    return patch_loader


