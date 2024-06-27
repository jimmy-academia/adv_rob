import sys
sys.path.append('.')
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision

from tqdm import tqdm
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

from utils import set_seeds, dumpj, loadj, check
from ipt.data import get_dataset, get_dataloader
from ipt.networks import DisjointPatchMaker, MLPTokenizer
from ipt.attacks import pgd_attack, square_attack


args = Namespace(batch_size=128, dataset='cifar10', seed=0)
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
        x = self.softmax(x * self.tau)
        x = x.permute(0, 2,3,1)
        x = x.view(x.size(0), -1, x.size(-1))
        x = torch.matmul(x, self.embedding.weight)
        x = self.patcher.inverse(x)
        return x
    
class Classifier(nn.Module):
    def __init__(self, args):
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
    def __init__(self, args, channel_list):
        super(Extention, self).__init__()
        _channel_list = []
        input_channel = args.channels
        for channel in channel_list:
            _channel_list.append(nn.Conv2d(input_channel, channel, kernel_size=3, stride=1, padding=1))
            _channel_list.append(nn.ReLU())
            input_channel = channel
        _channel_list.append(nn.Conv2d(input_channel, args.channels, kernel_size=3, stride=1, padding=1))
        self.layer = nn.Sequential(*_channel_list)
    
    def forward(self, x):
        x = self.layer(x)
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

    if args.do_softmax:
        model.tau = 0.1
    else:
        model.tau = 1

    Record = defaultdict(list)

    # model = IPTNet(args).to(args.device)
    classifier = Classifier(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt_class = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    
    elapsed_time = 0
    for iter_ in range(num_iter):
        start = time.time()    

        pbar = tqdm(train_loader, ncols=90, desc='IPT sim pretraining')
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
            pbar.set_postfix({'adv_acc': float(adv_acc), 'tau': model.tau})

        if args.do_softmax:
            model.tau *= 2

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

def get_vgg_model(num_classes=10, channels=3):
    model = torchvision.models.vgg11_bn(weights=None)
    model.features[0] = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model

def get_resnet_model(num_classes=10, channels=3):
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def main():

    ckpt_dir = Path('ckpt/1_more_compare')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    args.do_softmax = False
    num_iter = 5
    aptnet = APTNet(args)
    small = Extention(args, [18, 64])
    medium = Extention(args, [16, 64, 128])
    large = Extention(args, [16, 64, 128, 128])
    verylarge = Extention(args, [16, 64, 128, 256, 512])
    # classifier = Classifier(args)

    resnet18 = get_resnet_model(channels=args.channels)
    vgg11 = get_vgg_model(channels=args.channels)

    for name, _net in zip(['apt', 'small', 'medium', 'large', 'verylarge', 'resnet18', 'vgg11'], [aptnet, small, medium, large, verylarge, resnet18, vgg11]):
        print(f"Number of trainable parameters in {name}: {sum(p.numel() for p in _net.parameters() if p.requires_grad)}")
    
    ## test group
    model = aptnet.to(args.device)

    for do_softmax in [True, False]:
        args.do_softmax = do_softmax
        soft_fix = 'soft' if do_softmax else 'nosoft'
        model = aptnet.to(args.device)

        print(f' == aptnet+{soft_fix}, AST ==')
        path = ckpt_dir/f'/aptnet_{soft_fix}_AST.json'
        Record = rec_iptadvsim_training('pgd', num_iter, model)
        dumpj(Record, path)
    
    del model # release memory

    ## control group
    for name, _net in zip(['apt', 'small', 'medium', 'large', 'verylarge', 'resnet18', 'vgg11'], [aptnet, small, medium, large, verylarge, classifier, resnet18, vgg11]):
        if name not in ['resnet18', 'vgg11']:
            model = Dummy(_net, Classifier(args)).to(args.device)
        else:
            model = _net.to(args.device)
        
        print(f' == {name}, AT ==')
        path = ckpt_dir/f'/{name}_AT.json'
        Record = rec_adversarial_training('pgd', num_iter, model)
        dumpj(Record, path)
        del model # release memory


if __name__ == '__main__':
    main()