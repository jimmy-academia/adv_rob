import time
start = time.time()
import torch
from argparse import Namespace
from ipt.data import get_dataset, get_dataloader
from utils import set_seeds
args = Namespace(batch_size=128, dataset='mnist', seed=0)
args.device = torch.device("cuda:0")
args.channels = 1 if args.dataset == 'mnist' else 3
args.image_size = 224 if args.dataset == 'imagenet' else 32
    
args.patch_size = 2
args.patch_numel = args.channels * args.patch_size**2 
args.num_patches_width = args.image_size // args.patch_size

args.vocab_size = 12
args.eps = 0.3 if args.dataset == 'mnist' else 8/255
args.attack_iters = 100

set_seeds(args.seed)
train_set, test_set = get_dataset(args.dataset)
train_loader, test_loader = get_dataloader(train_set, test_set, args.batch_size)

import torch
import torch.nn as nn
from ipt.networks import DisjointPatchMaker, MLPTokenizer

class IPTNet(nn.Module):
    def __init__(self, args):
        super(IPTNet, self).__init__()
        self.args = args
        self.patcher = DisjointPatchMaker(args)
        self.patch_numel = args.channels * args.patch_size * args.patch_size
        self.tokenizer = MLPTokenizer(args.patch_numel, args.vocab_size, [64, 128])
        self.embedding = nn.Embedding(args.vocab_size, self.patch_numel)
        self.conv1 = nn.Conv2d(args.channels, 64, kernel_size=3, stride=1, padding=1)
        self.softmax = nn.Softmax(-1)
        self.tau = 1

    def forward(self, x):
        patches = self.patcher(x)
        # patches = patches.view(-1, self.patch_numel)
        x = self.tokenizer(patches)
        x = self.softmax(x * self.tau)
        x = torch.matmul(x, self.embedding.weight)
        x = self.patcher.inverse(x)
        return x

    def visualize_tok_image(self, img):
        x = self.patcher(img)  # (batch_size, num_patches, patch_numel)
        x = self.tokenizer(x) # (batch_size, num_patches, vocab_size)
        tok_image = x.argmax(2)
        tok_image = tok_image.view(self.args.num_patches_width, self.args.num_patches_width)
        print(tok_image)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(args.channels, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
    
class Dummy(nn.Module):
    def __init__(self, iptnet, classifier):
        super(Dummy, self).__init__()
        self.iptnet = iptnet
        self.classifier = classifier
    
    def forward(self, x):
        x = self.iptnet(x)
        x = self.classifier(x)
        return x


# from IPython.display import clear_output, display
from torchvision import transforms
import matplotlib.pyplot as plt
from ipt.attacks import pgd_attack, square_attack
from tqdm import tqdm
from pathlib import Path


def test_attack(args, model, test_loader, adv_perturb, fast=True):
    total = correct = adv_correct = 0
    model.eval()
    pbar = tqdm(test_loader, ncols=90, desc='test_attack', unit='batch', leave=False)
    for images, labels in pbar:
        images = images.to(args.device)
        labels = labels.to(args.device)
        pred = model(images)
        correct += (pred.argmax(dim=1) == labels).sum()

        adv_images = adv_perturb(args, images, model, labels)
        adv_pred = model(adv_images)
        adv_correct += (adv_pred.argmax(dim=1) == labels).sum()
        total += len(labels)

        if fast:
            break
    return correct, adv_correct, total



def show_image(image): 
    rootdir = Path('ckpt/temp_fig')
    id_ = len(list(rootdir.iterdir()))
    img = transforms.ToPILImage()(image.cpu().detach().squeeze())
    img.save(rootdir/f'image{id_}.jpg')

model = IPTNet(args).to(args.device)
classifier = Classifier().to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
opt_class = torch.optim.Adam(classifier.parameters(), lr=1e-3)

torch.autograd.set_detect_anomaly(True)

for iter_ in range(20):
    pbar = tqdm(train_loader, ncols=90, desc='IPT sim training')
    for images, labels in pbar:
        images = images.to(args.device)
        labels = labels.to(args.device)
        adv_images = pgd_attack(args, images, model, images, True)
        output = model(adv_images)
        mseloss = nn.MSELoss()(output, images)
        optimizer.zero_grad()
        mseloss.backward()
        optimizer.step()
        pbar.set_postfix(loss=mseloss.item())

    model.tau *= 1.2
    # print()
    # print(f' ====== Iter: {iter_}, Loss: {mseloss.item()} ======')
    # show_image(adv_images[0:1])
    # show_image(output[0:1])
    # torch.save(model.cpu().state_dict(), f'ckpt/ipt_mnist{}.pth')
    # model.to(args.device)

    total = correct = 0
    pbar = tqdm(train_loader, ncols=90, desc='classifier training')
    for images, labels in pbar:
        images = images.to(args.device)
        labels = labels.to(args.device)
        output = classifier(model(images))
        loss = nn.CrossEntropyLoss()(output, labels)
        opt_class.zero_grad()
        loss.backward()
        opt_class.step()
        total += labels.size(0)
        correct += (output.argmax(1) == labels).float().mean()
        acc = correct / total
        pbar.set_postfix(acc=acc)
    # print("classifier train accuracy:", correct / total)
    # print('== adversarial/sim training ==')
    pbar = tqdm(train_loader, ncols=88, desc='adversarial/sim training')
    dummy = Dummy(model, classifier)
    for images, labels in pbar:
        images, labels = images.to(args.device), labels.to(args.device)
        adv_images = square_attack(args, images, dummy, labels)
        # adv_images = pgd_attack(args, images, dummy, labels, False)
        output = model(adv_images)

        mseloss = nn.MSELoss()(output, images)
        optimizer.zero_grad()
        mseloss.backward(retain_graph=True)
        optimizer.step()

        cl_output = classifier(output.detach())
        cl_loss = nn.CrossEntropyLoss()(cl_output, labels)
        opt_class.zero_grad()
        cl_loss.backward()
        opt_class.step()

        adv_acc = (cl_output.argmax(dim=1) == labels).sum() / len(labels)
        pbar.set_postfix({'adv_acc': float(adv_acc)})
        # if pbar.n % (pbar.total//2)==0:
            # show_image(adv_images[0:1])
            # show_image(output[0:1])

    show_image(adv_images[0:1])
    show_image(output[0:1])
    

    correct, adv_correct, total = test_attack(args, dummy, test_loader, pgd_attack)
    message = f'[pgd attack] test acc: {correct/total:.4f}, adv acc: {adv_correct/total:.4f}...'
    print(message)

    correct, adv_correct, total = test_attack(args, dummy, test_loader, square_attack)
    message = f'[square attack] test acc: {correct/total:.4f}, adv acc: {adv_correct/total:.4f}...'
    print(message)
    torch.save(dummy.cpu().state_dict(), f'ckpt/square_train/ipt_both_{iter_}.pth')
    dummy.to(args.device)

    duration = time.time() - start
    if duration > 300:
        print(duration/60, 'minutes have past since start!')
    elif duration > 3600:
        print(duration//3600, 'hours and', duration % 3600, 'minutes have past since start!')
    else:
        print(duration, 'seconds have past since start!')

