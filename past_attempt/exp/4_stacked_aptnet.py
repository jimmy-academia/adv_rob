import sys 
sys.path.append('.')
import time
import torch 
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from functools import partial
from argparse import Namespace

from ipt.attacks import pgd_attack, auto_attack, square_attack
from ipt.networks import APTNet
from ipt.data import get_dataset, get_dataloader
from ipt.train import run_tests

from config import default_arguments
from utils import dumpj, loadj, check, debug_mode

class StackedAPTNet(nn.Module):
    def __init__(self, args):
        super(StackedAPTNet, self).__init__()
        self.apt1 = APTNet(args, [16, 32])
        self.conv1 = nn.Sequential(
            nn.Conv2d(args.channels, 8, 3, 1, 1),
            nn.Conv2d(8, 16, 3, 2, 1)
        ) 
        specs = Namespace(channels=16, image_size=args.image_size//2, patch_size=args.patch_size, vocab_size=args.vocab_size*2)
        self.apt2 = APTNet(specs, [32, 64])
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.Conv2d(32, 64, 3, 2, 1)
        ) 
        self.conv3 = nn.Conv2d(64, 64, 3, 2, 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(256, args.num_classes) # 256 = 64 * 2 * 2
    
    def forward(self, x):
        x = self.apt1(x)
        x = self.after1(x)
        return x
    
    def after1(self, x):
        x = self.conv1(x)
        x = self.apt2(x)
        x = self.after2(x)
        return x

    def before2(self, x):
        x = self.apt1(x)
        x = self.conv1(x)
        return x
    
    def with2(self, x):
        x = self.before2(x)
        x = self.apt2(x)
        return x

    def after2(self, x):
        x = self.conv2(x) # 64x8x8
        x = self.pool(x) # 64x4x4
        x = self.conv3(x) # 64x2x2
        x = self.flatten(x) # 256
        x = self.linear(x)
        return x


def do_adversarial_similarity_training(args, model, train_loader, test_loader, adv_attacks, atk_names):
    # assume: model.iptnet vs model.classifier
    Record = defaultdict(list)

    opt_apt1 = torch.optim.Adam(model.apt1.parameters(), lr=1e-3)
    opt_apt2 = torch.optim.Adam(model.apt2.parameters(), lr=1e-3)
    all_params = set(model.parameters())
    apt1_params = set(model.apt1.parameters())
    apt2_params = set(model.apt2.parameters())
    not1_params = all_params - apt1_params
    not12_params = all_params - apt1_params - apt2_params

    opt_not1 = torch.optim.Adam(not1_params, lr=1e-3)
    opt_not12 = torch.optim.Adam(not12_params, lr=1e-3)

    elapsed_time = 0
    for epoch in range(args.num_epochs):
        start = time.time()    
        pbar = tqdm(train_loader, ncols=90, desc='ast:pretrain')
        for images, labels in pbar:
            images = images.to(args.device)
            labels = labels.to(args.device)
            adv_images = pgd_attack(args, images, model.apt1, images, True)
            output = model.apt1(adv_images)
            mseloss = nn.MSELoss()(output, images)
            opt_apt1.zero_grad()
            mseloss.backward()
            opt_apt1.step()
            pbar.set_postfix(loss=mseloss.item())

        pbar = tqdm(train_loader, ncols=88, desc='adv/sim training')
        
        for images, labels in pbar:
            images, labels = images.to(args.device), labels.to(args.device)
            adv_images = pgd_attack(args, images, model, labels, False)
            output = model.apt1(adv_images)

            mseloss = nn.MSELoss()(output, images)
            opt_apt1.zero_grad()
            mseloss.backward()
            opt_apt1.step()

            cl_output = model.after1(output.detach())
            cl_loss = nn.CrossEntropyLoss()(cl_output, labels)
            opt_not1.zero_grad()
            cl_loss.backward()
            opt_not1.step()

            adv_acc = (cl_output.argmax(dim=1) == labels).sum() / len(labels)
            pbar.set_postfix({'adv_acc': float(adv_acc)})

        pbar = tqdm(train_loader, ncols=90, desc='ast:pretrain2')
        for images, labels in pbar:
            images = images.to(args.device)
            labels = labels.to(args.device)
            standard = model.before2(images).detach()
            adv_images = pgd_attack(args, images, model.with2, standard, True)
            output = model.with2(adv_images)
            mseloss = nn.MSELoss()(output, standard)
            opt_apt2.zero_grad()
            mseloss.backward()
            opt_apt2.step()
            pbar.set_postfix(loss=mseloss.item())

        pbar = tqdm(train_loader, ncols=88, desc='adv/sim training')
        
        for images, labels in pbar:
            images, labels = images.to(args.device), labels.to(args.device)
            adv_images = pgd_attack(args, images, model, labels, False)
            standard = model.before2(images).detach()
            output = model.with2(adv_images)

            mseloss = nn.MSELoss()(output, standard)
            opt_apt2.zero_grad()
            mseloss.backward()
            opt_apt2.step()

            cl_output = model.after2(output.detach())
            cl_loss = nn.CrossEntropyLoss()(cl_output, labels)
            opt_not12.zero_grad()
            cl_loss.backward()
            opt_not12.step()

            adv_acc = (cl_output.argmax(dim=1) == labels).sum() / len(labels)
            pbar.set_postfix({'adv_acc': float(adv_acc)})
    

        elapsed_time += time.time() - start
        Record['epoch'].append(epoch)
        Record['elapsed_time'].append(elapsed_time)
        Record = run_tests(args, model, test_loader, Record, adv_attacks, atk_names)
        
    return Record

def main():
    args = default_arguments('mnist')
    args.num_epochs = 20
    args.ckpt_dir = Path('ckpt/4_stacked_aptnet')
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_set, test_set = get_dataset(args.dataset)
    train_loader, test_loader = get_dataloader(train_set, test_set, args.batch_size)
    
    record_path = args.ckpt_dir / 'apt_ast.json'
    auto_attack_rand = partial(auto_attack, _version='plus')

    model = StackedAPTNet(args).to(args.device)
    print(f"Number of trainable parameters in apt model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
    ast_record = do_adversarial_similarity_training(args, model, train_loader, test_loader, [pgd_attack, square_attack, auto_attack_rand], ['pgd', 'square', 'autoplus'])
    dumpj(ast_record, record_path)
    print('Done!')

if __name__ == '__main__':
    debug_mode()
    main()

    # model = BasicBlock(3, 16, 2)
    # x = torch.randn(1, 3, 32, 32)
    # print(model(x).shape)
