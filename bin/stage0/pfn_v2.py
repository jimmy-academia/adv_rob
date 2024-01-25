import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

input('>>>>> not avliable <<<<<<')

num_epochs = 128
modelpath = Path('ckpt/pfnv1.pth')

class PFN_v2(nn.Module):
    """docstring for PFN_v2"""
    def __init__(self):
        super(PFN_v2, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

        self.protos = nn.Embedding(64, 128)
        self.tau = 0.1

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)

        # B, 128 => B, 128
        sim = torch.matmul(x, self.protos.weight.T)
        ssim = F.softmax(self.tau*sim, dim=1)
        x = torch.matmul(ssim, self.protos.weight)

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
    print('create non-confusable features prototypes!?')

    traindata = torchvision.datasets.MNIST(DSETROOT, transform=TRANS, download=True)
    testdata = torchvision.datasets.MNIST(DSETROOT, transform=TRANS, download=True, train=False)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=128, shuffle=True, num_workers = 8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testdata, batch_size=128, shuffle=False, num_workers = 8, pin_memory=True)

    model = PFN_v2()
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
            model.tau *= 1.03

        print('Training finished!')
        torch.save(model.state_dict(), modelpath)
    else:
        model.load_state_dict(torch.load(modelpath))
    
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
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
    # epsilon = 0.1
    epsilon = 0.5
    # model.tau = 0.1 * 1.03 ** 128
    # print(model.tau)

    alpha = 0.01
    num_iter = 100
    for imgs, labels in testloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        model.tau = 0.1
        adv_imgs = pgd_attack(model, imgs, labels, epsilon, alpha, num_iter)

        # transforms.ToPILImage()(adv_imgs[0]).save('sample.jpg')
        # input('save_sample')
        model.tau = 0.1 * 1.03 ** 128
        outputs = model(adv_imgs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Adversarial Accuracy: {accuracy * 100:.2f}%')



if __name__ == '__main__':
    main()