import torch
import torch.nn as nn
from tqdm import tqdm 

def pgd_attack(args, model, images, labels):
    images = images.clone().detach().requires_grad_(True)
    for _ in range(args.num_iter):
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        grad = torch.autograd.grad(loss, images)[0]

        # Add perturbation with epsilon and clip within [0, 1]
        images = images + args.alpha * torch.sign(grad)
        images = torch.clamp(images, 0, 1)

        # Project the perturbed images to the epsilon ball around the original images
        images = torch.max(torch.min(images, images + args.epsilon), images - args.epsilon)
        images = torch.clamp(images, 0, 1)

    return images.detach()

def exp_test(args, model, testloader, device=0):
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

        adv_imgs = pgd_attack(args, model, imgs, labels)
        outputs = model(adv_imgs)
        _, predicted = torch.max(outputs.data, 1)
        atotal += labels.size(0)
        acorrect += (predicted == labels).sum().item()

    testacc = correct / total 
    attackacc = acorrect / atotal
    return testacc, attackacc

def exp_transfer(args, model, Nmod, testloader, device=0):

    correct = total = 0
    pbar = tqdm(testloader, ncols=88, desc='test', leave=False)

    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        adv_imgs = pgd_attack(args, Nmod, imgs, labels)
        outputs = model(adv_imgs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = correct / total
    return acc
