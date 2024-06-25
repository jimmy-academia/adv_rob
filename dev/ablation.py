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


class APTNet(nn.Module):
    def __init__(self, args):
        super(APTNet, self).__init__()
        self.args = args
        self.patcher = DisjointPatchMaker(args)
        self.patch_numel = args.channels * args.patch_size * args.patch_size
        self.conv1 = nn.Conv2d(args.channels, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, args.vocab_size, kernel_size=1, stride=1)
        self.embedding = nn.Embedding(args.vocab_size, self.patch_numel)
        self.softmax = nn.Softmax(-1)
        self.relu = nn.ReLU()
        self.tau = 1

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        if args.do_softmax:
            x = self.softmax(x * self.tau)
        x = x.permute(0, 2,3,1)
        x = x.view(x.size(0), -1, x.size(-1))
        x = torch.matmul(x, self.embedding.weight)
        x = self.patcher.inverse(x)
        return x

class IPTNet(nn.Module):
    def __init__(self, args):
        super(IPTNet, self).__init__()
        self.args = args
        self.patcher = DisjointPatchMaker(args)
        self.patch_numel = args.channels * args.patch_size * args.patch_size
        self.tokenizer = MLPTokenizer(args.patch_numel, args.vocab_size, [64, 128])
        self.embedding = nn.Embedding(args.vocab_size, self.patch_numel)
        self.softmax = nn.Softmax(-1)
        self.tau = 1

    def forward(self, x):
        patches = self.patcher(x)
        x = self.tokenizer(patches)
        if args.do_softmax:
            x = self.softmax(x * self.tau)
        x = torch.matmul(x, self.embedding.weight)
        x = self.patcher.inverse(x)
        return x


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
    
class Extention(nn.Module):
    def __init__(self, args):
        super(Extention, self).__init__()
        self.conv1 = nn.Conv2d(args.channels, 18, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(18, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, args.channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
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


def rec_iptadvsim_training(adv_train_type, num_iter, model):

    Record = defaultdict(list)

    # model = IPTNet(args).to(args.device)
    classifier = Classifier().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt_class = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    
    elapsed_time = 0
    for iter_ in range(num_iter):
        start = time.time()    
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
        message = f'[AST pgd] test acc: {correct/total:.4f}, adv acc: {adv_correct/total:.4f}...'
        print(message)
        Record['pgd_test_acc'].append(correct/total)
        Record['pgd_adv_acc'].append(adv_correct/total)

        correct, adv_correct, total = test_attack(args, dummy, test_loader, square_attack)
        message = f'[AST square] test acc: {correct/total:.4f}, adv acc: {adv_correct/total:.4f}...'
        print(message)
        Record['square_test_acc'].append(correct/total)
        Record['square_adv_acc'].append(adv_correct/total)
    return Record

def rec_adversarial_training(adv_train_type, num_iter, model, tau=1):
    Record = defaultdict(list)
    # model = IPTNet(args).to(args.device)
    # model.tau = tau
    # classifier = Classifier().to(args.device)
    # dummy = Dummy(model, classifier)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    elapsed_time = 0
    for iter_ in range(num_iter):
        start = time.time()
        pbar = tqdm(train_loader, ncols=90, desc='adversarial training')
        for images, labels in pbar:
            images = images.to(args.device)
            labels = labels.to(args.device)
            if adv_train_type == 'pgd':
                adv_images = pgd_attack(args, images, model, labels, False)
            elif adv_train_type == 'square':
                adv_images = square_attack(args, images, model, labels)
            output = model(adv_images)
            loss = nn.CrossEntropyLoss()(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            adv_acc = (output.argmax(dim=1) == labels).sum() / len(labels)
            pbar.set_postfix({'adv_acc': float(adv_acc)})

        elapsed_time += time.time() - start
        Record['iter'].append(iter_)
        Record['elapsed_time'].append(elapsed_time)

        correct, adv_correct, total = test_attack(args, model, test_loader, pgd_attack)
        Record['pgd_test_acc'].append(correct/total)
        Record['pgd_adv_acc'].append(adv_correct/total)
        message = f'[AT pgd] test acc: {correct/total:.4f}, adv acc: {adv_correct/total:.4f}...'
        print(message)
        
        correct, adv_correct, total = test_attack(args, model, test_loader, square_attack)
        message = f'[AT square] test acc: {correct/total:.4f}, adv acc: {adv_correct/total:.4f}...'
        print(message)
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
    num_iter = 5
    aptnet = APTNet(args).to(args.device)
    iptnet = IPTNet(args).to(args.device)
    ext = Extention(args).to(args.device)
    args.do_softmax = True
    testimg = torch.rand(3,1,32,32).to(args.device)
    print(f"APTN: {aptnet(testimg).shape}")
    print(f"IPTN: {iptnet(testimg).shape}")
    print(f"Extention: {ext(testimg).shape}")

    print(f"Number of trainable parameters in APTNet: {sum(p.numel() for p in aptnet.parameters() if p.requires_grad)}")
    print(f"Number of trainable parameters in IPTNet: {sum(p.numel() for p in iptnet.parameters() if p.requires_grad)}")
    print(f"Number of trainable parameters in Extention: {sum(p.numel() for p in ext.parameters() if p.requires_grad)}")
    
    # input('pause... check')
    ## AT
    name = 'normal'
    model = Dummy(ext, Classifier()).to(args.device)
    for adv_train_type in ['pgd', 'square']:
        print(f' == {name}, AT_{adv_train_type} ==')
        path = Path(f'ckpt/ablation/{name}_AT_{adv_train_type}.json')
        Record = rec_adversarial_training(adv_train_type, num_iter, model)
        dumpj(Record, path)


    ## AST
    for _net, name in zip([iptnet, aptnet], ['ipt', 'apt']):
        for do_softmax in [True, False]:
            args.do_softmax = do_softmax
            for adv_train_type in ['pgd', 'square']:
                soft_fix = 'soft' if do_softmax else 'nosoft'

                print(f' == {name}+{soft_fix}, AST_{adv_train_type} ==')

                path = Path(f'ckpt/ablation/{name}_{soft_fix}_AST_{adv_train_type}.json')

                Record = rec_iptadvsim_training(adv_train_type, num_iter, _net)
                dumpj(Record, path)

    for _net, name in zip([iptnet, aptnet], ['ipt', 'apt']):
        model = Dummy(_net, Classifier()).to(args.device)
        for do_softmax in [True, False]:
            args.do_softmax = do_softmax
            for adv_train_type in ['pgd', 'square']:
                soft_fix = 'soft' if do_softmax else 'nosoft'

                print(f' == {name}+{soft_fix}, AT_{adv_train_type} ==')

                path = Path(f'ckpt/ablation/{name}_{soft_fix}_AT_{adv_train_type}.json')
                Record = rec_adversarial_training(adv_train_type, num_iter, model)

                dumpj(Record, path)




if __name__ == '__main__':
    main()