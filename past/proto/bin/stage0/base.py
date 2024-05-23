import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from utils import *

from pathlib import Path

DSETROOT = '../DATASET/'
TRANS = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,))  
])
device = torch.device("cuda:"+str(0))
# torch.device("cpu" if args.device == -1 else "cuda:"+str(args.device))

num_epochs = 128
modelpath = Path('ckpt/test.pth')

class BaseNet(nn.Module):
    """docstring for BaseNet"""
    def __init__(self):
        super(BaseNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def pgd_attack(model, images, labels, epsilon, alpha, num_iter):
    images = images.clone().detach().requires_grad_(True)
    for _ in range(num_iter):
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        grad = torch.autograd.grad(loss, images)[0]

        # Add perturbation with epsilon and clip within [0, 1]
        images = images + alpha * torch.sign(grad)
        images = torch.clamp(images, 0, 1)

        # Project the perturbed images to the epsilon ball around the original images
        images = torch.max(torch.min(images, images + epsilon), images - epsilon)
        images = torch.clamp(images, 0, 1)

    return images.detach()
            

def main():
    print('make a Basic network and check coding')



    traindata = torchvision.datasets.MNIST(DSETROOT, transform=TRANS, download=True)
    testdata = torchvision.datasets.MNIST(DSETROOT, transform=TRANS, download=True, train=False)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=128, shuffle=True, num_workers = 8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testdata, batch_size=128, shuffle=False, num_workers = 8, pin_memory=True)

    model = BaseNet()
    model.to(device)

    if not modelpath.exists():
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # training
        for epoch in range(num_epochs):
            running_loss = 0.0
            for imgs, labels in trainloader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(trainloader)}')

        print('Training finished!')
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0

        torch.save(model.state_dict(), modelpath)
    else:
        model.load_state_dict(torch.load(modelpath))

    with torch.no_grad():
        for imgs, labels in testloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    correct = 0
    total = 0
    epsilon = 0.1
    alpha = 0.01
    num_iter = 10
    for imgs, labels in testloader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        adv_imgs = pgd_attack(model, imgs, labels, epsilon, alpha, num_iter)

        outputs = model(adv_imgs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Adversarial Accuracy: {accuracy * 100:.2f}%')



if __name__ == '__main__':
    main()