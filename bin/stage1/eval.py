import torch
import torch.nn as nn
from tqdm import tqdm 

def pgd_attack(model, images, labels):
    num_iter = 20
    alpha = 0.01
    epsilon = 0.03
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

def exp_transfer(model, Nmod, testloader, device=0):

    correct = total = 0
    pbar = tqdm(testloader, ncols=88, desc='test', leave=False)

    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        adv_imgs = pgd_attack(Nmod, imgs, labels)
        outputs = model(adv_imgs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = correct / total
    return acc
