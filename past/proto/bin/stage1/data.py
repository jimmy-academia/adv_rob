import torch
import torchvision
from torchvision import transforms
DSETROOT = '../../../DATASET/'
TRANS = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,))  
])


def get_dataloader(dataset='mnist', batch_size=128):

    match dataset:
        case 'mnist':
            traindata = torchvision.datasets.MNIST(DSETROOT, transform=TRANS, download=True)
            testdata = torchvision.datasets.MNIST(DSETROOT, transform=TRANS, download=True, train=False)
            trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers = 8, pin_memory=True)
            testloader = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers = 8, pin_memory=True)

            return trainloader, testloader

        case _:
            print(dataset, 'dataset not defined!')
