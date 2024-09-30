import sys 
sys.path.append('.')

import time
import torch
from pathlib import Path
import torch.nn as nn
from tqdm import tqdm
from argparse import Namespace
from functools import partial
from collections import defaultdict

from config import default_arguments
from ipt.data import get_dataset, get_dataloader
from ipt.attacks import pgd_attack, auto_attack
from ipt.train import run_tests
from ipt.networks import APTNet
from utils import dumpj, check, debug_mode

import torch
import torch.nn as nn
from types import SimpleNamespace


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class MultiStackedAPTNet(nn.Module):
    def __init__(self, args):
        super(MultiStackedAPTNet, self).__init__()
        self.num_aptnets = args.num_aptnets
        self.aptnets = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()
        
        current_channels = args.channels
        current_size = args.image_size
        
        for i in range(args.num_aptnets):
            # Create APTNet
            next_size = current_channels * 2 if i != 0 else 8
            hidden_sizes = [current_channels, next_size]
            self.aptnets.append(APTNet(SimpleNamespace(channels=current_channels, 
                                                       image_size=current_size, 
                                                       patch_size=args.patch_size, 
                                                       vocab_size=args.vocab_size * 2**i), 
                                       hidden_sizes))
            
            # Create Residual Block
            s = 2 if i<=2 else 1
            self.residual_blocks.append(ResidualBlock(current_channels, hidden_sizes[-1], stride=s))
            
            current_channels = hidden_sizes[-1]
            current_size //= s
        
        self.final_conv = nn.Conv2d(current_channels, current_channels*2, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d((2, 2))
        self.flatten = nn.Flatten()

        final_neural_num = current_channels*2 * (current_size//4) **2
        self.linear = nn.Linear(final_neural_num, args.num_classes)
        self.generate_methods()
    
    def forward(self, x):
        x = self.aptnets[0](x)
        return self.after1(x)
    
    def generate_methods(self):
        for i in range(2, self.num_aptnets+1):
            setattr(self, f'before{i}', self._generate_before_method(i))
            setattr(self, f'with{i}', self._generate_with_method(i)) 
        for i in range(self.num_aptnets, 0, -1):
            setattr(self, f'after{i}', self._generate_after_method(i))
    
    def _generate_before_method(self, n):
        def before_n(x):
            if n > 2:
                x = getattr(self, f'before{n-1}')(x)
            x = self.aptnets[n-2](x)
            x = self.residual_blocks[n-2](x)
            return x
        return before_n
    
    def _generate_with_method(self, n):
        def with_n(x):
            x = getattr(self, f'before{n}')(x)
            x = self.aptnets[n-1](x)
            return x
        return with_n
    
    def _generate_after_method(self, n):
        def after_n(x):
            x = self.residual_blocks[n-1](x)
            if n < self.num_aptnets:
                x = self.aptnets[n](x)
                x = getattr(self, f'after{n+1}')(x)
                return x
            else:
                x = self.final_conv(x)
                x = self.pool(x)
                x = self.flatten(x)
                x = self.linear(x)
                return x
        return after_n


def do_adversarial_similarity_training(args, model, train_loader, test_loader, adv_attacks, atk_names):
    Record = defaultdict(list)
    
    normal_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Create optimizers for each APTNet and the rest of the model
    optimizers = []
    for i in range(model.num_aptnets):
        optimizers.append(torch.optim.Adam(model.aptnets[i].parameters(), lr=1e-3))


    # Create optimizers for the rest of the model parameters
    all_params = set(model.parameters())
    apt_params = set()
    for i in range(model.num_aptnets):
        apt_params.update(set(model.aptnets[i].parameters()))
    
    not_apt_params = all_params - apt_params
    opt_not_apt = torch.optim.Adam(not_apt_params, lr=1e-3)

    elapsed_time = 0
    for epoch in range(args.num_epochs):
        start = time.time()
        pbar = tqdm(train_loader, ncols=90, desc=f'normal training {epoch}')
        model.train()
        for images, labels in pbar:
            images = images.to(args.device)
            labels = labels.to(args.device)
            output = model(images)
            loss = nn.CrossEntropyLoss()(output, labels)
            normal_optimizer.zero_grad()
            loss.backward()
            normal_optimizer.step()
            pbar.set_postfix(loss=float(loss.item()), acc=float((output.argmax(dim=1) == labels).sum() / len(labels)))

        for n in range(model.num_aptnets):
            # Pretraining phase
            pbar = tqdm(train_loader, ncols=90, desc=f'ast:pretrain{n+1}')
            for images, labels in pbar:
                images = images.to(args.device)
                labels = labels.to(args.device)
                
                if n == 0:
                    standard = images
                    adv_images = pgd_attack(args, images, model.aptnets[n], standard, True)
                    output = model.aptnets[n](adv_images)
                else:
                    standard = getattr(model, f'before{n+1}')(images).detach()
                    adv_images = pgd_attack(args, images, getattr(model, f'with{n+1}'), standard, True)
                    output = getattr(model, f'with{n+1}')(adv_images)
                mseloss = nn.MSELoss()(output, standard)
                
                optimizers[n].zero_grad()
                mseloss.backward()
                optimizers[n].step()
                
                pbar.set_postfix(loss=mseloss.item())

            # Adversarial training phase
            pbar = tqdm(train_loader, ncols=88, desc=f'adv/sim training {n+1}')
            for images, labels in pbar:
                images, labels = images.to(args.device), labels.to(args.device)
                adv_images = pgd_attack(args, images, model, labels, False)
                
                if n == 0:
                    standard = images
                    output = model.aptnets[n](adv_images)
                else:
                    standard = getattr(model, f'before{n+1}')(images).detach()
                    output = getattr(model, f'with{n+1}')(adv_images)
                mseloss = nn.MSELoss()(output, standard)
                
                optimizers[n].zero_grad()
                mseloss.backward()
                optimizers[n].step()
                
                cl_output = getattr(model, f'after{n+1}')(output.detach())
                cl_loss = nn.CrossEntropyLoss()(cl_output, labels)
                
                opt_not_apt.zero_grad()
                cl_loss.backward()
                opt_not_apt.step()
                
                adv_acc = (cl_output.argmax(dim=1) == labels).sum() / len(labels)
                pbar.set_postfix({'adv_acc': float(adv_acc)})

        elapsed_time += time.time() - start
        Record['epoch'].append(epoch)
        Record['elapsed_time'].append(elapsed_time)
        Record = run_tests(args, model, test_loader, Record, adv_attacks, atk_names)
    
    return Record

def main():
    debug_mode()
    args = default_arguments('cifar10')
    args.num_epochs = 5
    args.vocab_size = 32
    for num_aptnets in range(1,10):
        print(f'==== Number of APTNets: {num_aptnets} =====')
        args.num_aptnets = num_aptnets
        args.ckpt_dir = Path('ckpt/4d_multi_train')
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)

        train_set, test_set = get_dataset(args.dataset)
        train_loader, test_loader = get_dataloader(train_set, test_set, args.batch_size)

        record_path = args.ckpt_dir / f'apt{num_aptnets}_result.json'
        auto_attack_rand = partial(auto_attack, _version='plus')

        model = MultiStackedAPTNet(args).to(args.device)
        print(f"Number of trainable parameters in apt model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        ast_record = do_adversarial_similarity_training(args, model, train_loader, test_loader, [pgd_attack, auto_attack_rand], ['pgd', 'autoplus'])
        dumpj(ast_record, record_path)
        print('Done!')


if __name__ == '__main__':
    main()

    # print(model(torch.rand(5, 3, 32, 32).to(args.device)).shape)
    # print('!!!!!!!!!')
