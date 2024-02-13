import torch
import torch.nn as nn

from data import get_dataloader

from tqdm import tqdm

def pgd_attack(model, images, labels):
    images = images.clone().detach().requires_grad_(True)
    for _ in range(20):
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        grad = torch.autograd.grad(loss, images)[0]

        # Add perturbation with epsilon and clip within [0, 1]
        images = images + 0.01 * torch.sign(grad)
        images = torch.clamp(images, 0, 1)

        # Project the perturbed images to the epsilon ball around the original images
        images = torch.max(torch.min(images, images + 0.03), images - 0.03)
        images = torch.clamp(images, 0, 1)

    return images.detach()

def exp_test(model, testloader, device=0):
    correct = total = 0
    acorrect = atotal = 0
    pbar = tqdm(testloader, ncols=88, desc='test', leave=False)

    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        adv_imgs = pgd_attack(model, imgs, labels)
        outputs = model(adv_imgs)
        _, predicted = torch.max(outputs.data, 1)
        atotal += labels.size(0)
        acorrect += (predicted == labels).sum().item()

    testacc = correct / total 
    attackacc = acorrect / atotal
    return testacc, attackacc

def main():
    trainloader, testloader = get_dataloader()
    model = nn.Sequential(
            nn.Conv2d(1, 16, 3,1,1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(16, 32, 3,1,1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 64, 3,1,1),
            nn.ReLU(), 
            nn.Flatten(),
            nn.Linear(64*7*7, 10),
        )

    model.to(0)

    for epoch in range(50):
        
        pbar = tqdm(trainloader, ncols=88, desc='train', leave=False)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        correct = total = 0

        for imgs, labels in pbar:
            imgs = imgs.to(0)
            labels = labels.to(0)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = correct / total
            pbar.set_postfix(acc=accuracy)
        
        testacc, attackacc = exp_test(model, testloader)
        print(epoch, testacc, attackacc)


if __name__ == '__main__':
    main()