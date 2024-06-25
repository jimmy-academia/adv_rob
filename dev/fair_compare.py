import sys
sys.path.append('.')
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

from utils import set_seeds, dumpj, loadj, check
from ipt.data import get_dataset, get_dataloader
from ipt.networks import DisjointPatchMaker, MLPTokenizer
from ipt.attacks import pgd_attack, square_attack


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
    

def test_attack(args, model, test_loader, adv_perturb, fast=True):
    total = correct = adv_correct = 0
    model.eval()
    pbar = tqdm(test_loader, ncols=90, desc='test_attack', unit='batch', leave=False)
    for images, labels in pbar:
        images = images.to(args.device)
        labels = labels.to(args.device)
        pred = model(images)
        correct += float((pred.argmax(dim=1) == labels).sum())

        adv_images = adv_perturb(args, images, model, labels)
        adv_pred = model(adv_images)
        adv_correct += float((adv_pred.argmax(dim=1) == labels).sum())
        total += len(labels)

        if fast:
            break
    return correct, adv_correct, total


def rec_iptadvsim_training(adv_train_type, num_iter = 20):

    Record = defaultdict(list)

    model = IPTNet(args).to(args.device)
    classifier = Classifier().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt_class = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    
    elapsed_time = 0
    for iter_ in range(num_iter):
        start = time.time()
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
            correct += (output.argmax(1) == labels).float().sum()
            acc = float(correct / total)
            pbar.set_postfix(acc=acc)

        pbar = tqdm(train_loader, ncols=88, desc='adversarial/sim training')
        dummy = Dummy(model, classifier)
        for images, labels in pbar:
            images, labels = images.to(args.device), labels.to(args.device)

            if adv_train_type == 'pgd':
                adv_images = pgd_attack(args, images, dummy, labels, False)
            elif adv_train_type == 'square':
                adv_images = square_attack(args, images, dummy, labels)
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

        elapsed_time += time.time() - start
        Record['iter'].append(iter_)
        Record['elapsed_time'].append(elapsed_time)

        correct, adv_correct, total = test_attack(args, dummy, test_loader, pgd_attack)
        # message = f'[pgd attack] test acc: {correct/total:.4f}, adv acc: {adv_correct/total:.4f}...'
        # print(message)
        Record['pgd_test_acc'].append(correct/total)
        Record['pgd_adv_acc'].append(adv_correct/total)

        correct, adv_correct, total = test_attack(args, dummy, test_loader, square_attack)
        Record['square_test_acc'].append(correct/total)
        Record['square_adv_acc'].append(adv_correct/total)
        # message = f'[square attack] test acc: {correct/total:.4f}, adv acc: {adv_correct/total:.4f}...'
        # print(message)
        # torch.save(dummy.cpu().state_dict(), f'ckpt/square_train/ipt_both_{iter_}.pth')
        # dummy.to(args.device)
    return Record

def rec_adversarial_training(adv_train_type, tau=1, num_iter=20):
    Record = defaultdict(list)
    model = IPTNet(args).to(args.device)
    model.tau = tau
    classifier = Classifier().to(args.device)
    dummy = Dummy(model, classifier)
    optimizer = torch.optim.Adam(dummy.parameters(), lr=1e-3)
    
    elapsed_time = 0
    for iter_ in range(num_iter):
        start = time.time()
        pbar = tqdm(train_loader, ncols=90, desc='adversarial training')
        for images, labels in pbar:
            images = images.to(args.device)
            labels = labels.to(args.device)
            if adv_train_type == 'pgd':
                adv_images = pgd_attack(args, images, dummy, labels, False)
            elif adv_train_type == 'square':
                adv_images = square_attack(args, images, dummy, labels)
            output = dummy(adv_images)
            loss = nn.CrossEntropyLoss()(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            adv_acc = (output.argmax(dim=1) == labels).sum() / len(labels)
            pbar.set_postfix({'adv_acc': float(adv_acc)})

        elapsed_time += time.time() - start
        Record['iter'].append(iter_)
        Record['elapsed_time'].append(elapsed_time)

        correct, adv_correct, total = test_attack(args, dummy, test_loader, pgd_attack)
        Record['pgd_test_acc'].append(correct/total)
        Record['pgd_adv_acc'].append(adv_correct/total)

        correct, adv_correct, total = test_attack(args, dummy, test_loader, square_attack)
        Record['square_test_acc'].append(correct/total)
        Record['square_adv_acc'].append(adv_correct/total)
    return Record

def plot_accuracies(records, title, key, ylabel, filename):
    plt.figure(figsize=(10, 6))
    for name, record in records:
        plt.plot(record['elapsed_time'], record[key], label=name)
    plt.xlabel('Elapsed Time (s)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(f'ckpt/fair_fig/{filename}.jpg')
    plt.close()

def main():
    num_iter = 40
    records = []
    for adv_train_type in ['pgd', 'square']:
        for exp_name, func in zip(['ipt_train', 'adv_train', 'adv_train_tau40'], [rec_iptadvsim_training, rec_adversarial_training, rec_adversarial_training]):

            path = Path(f'ckpt/fair/{adv_train_type}_{exp_name}.json')
            if path.exists():
                Record = loadj(path)
            else:
                if 'tau' in exp_name:
                    Record = func(adv_train_type, num_iter=num_iter, tau=40)
                else:
                    Record = func(adv_train_type, num_iter=num_iter)
                dumpj(Record, path)

            records.append((f'{exp_name}_{adv_train_type}', Record))
    
    plot_accuracies(records, 'IPT vs Adversarial Training', 'pgd_test_acc', 'Test Accuracy', 'test_acc')
    plot_accuracies(records, 'IPT vs Adversarial Training', 'pgd_adv_acc', 'Adversarial Accuracy', 'adv_acc')    
    plot_accuracies(records, 'IPT vs Adversarial Training', 'square_test_acc', 'Test Accuracy', 'stest_acc')
    plot_accuracies(records, 'IPT vs Adversarial Training', 'square_adv_acc', 'Adversarial Accuracy', 'sadv_acc')    



    

if __name__ == '__main__':
    main()