import torch
import torchvision
import torchvision.transforms as transforms

from utils import readf, check

from tqdm import tqdm

# Rootdir = '/home/jimmyyeh/Documents/CRATER/DATASET'
Rootdir = readf('ckpt/rootdir')
print(Rootdir)
check()
# Import MNIST dataset

def get_dataset(name='mnist'):
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
def get_dataloader(trainset, testset, batch_size=64):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

def split_patch(images, patch_size):
    return images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).contiguous().view(images.size(0), -1, patch_size*patch_size*images.size(1)).float()
    
def tokenize_dataset(_loader, tokenizer, patch_size, device=0):
    tok_set = []
    for images, labels in tqdm(_loader, ncols=90, desc='tokenizing', unit='images', leave=False):
        images = images.to(device)
        patches = split_patch(images, patch_size)
        token_probability = tokenizer(patches)
        tok_images = torch.argmax(token_probability, dim=2)  # Assign the largest element in tok_image to tok_image
        width = images.shape[-1]//patch_size # Assume square images
        tok_images = tok_images.view(images.size(0), width, width)
        tok_images = tok_images.to('cpu')
        
        tok_set.extend([(tok_images[i], labels[i].item()) for i in range(tok_images.size(0))])
    return tok_set
