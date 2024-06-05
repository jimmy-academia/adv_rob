import torch
from parse import set_arguments
from utils import set_seeds
from data import get_dataset, get_dataloader, tokenize_dataset, patchwise_loader
from networks import IPTResnet
from attack import adv_perturb
from train import train_classifier

from tqdm import tqdm

import random


args = set_arguments()
args.dataset = 'cifar10'
args.vocab_size = 10240
args.batch_size = 128
args.channels = 3
set_seeds(args.seed)
train_set, test_set = get_dataset(args.dataset)
N = len(test_set)
reduced_test_set = [test_set[i] for i in random.sample(range(N), N//5)]
train_loader, test_loader = get_dataloader(train_set, reduced_test_set, args.batch_size)

iptresnet = IPTResnet(args).to(args.device)

pbar = tqdm(range(args.toktrain_epochs), ncols=90, desc='advtr. patch. testing...')


for epoch in pbar:
    print()
    print(f'========={epoch}=========')
    correct = total = 0
    tok_file = args.ckpt_dir/f'{args.vocab_size}/tok{epoch}.pth'
    if tok_file.exists():
        iptresnet.tokenizer.load_state_dict(torch.load(tok_file))
    else:
        break

    # for images, __ in tqdm(test_loader, ncols=70, desc='test iptresnet.tokenizer', leave=False):
    #     images = images.to(args.device)
    #     patches = images.view(-1, args.patch_size)
    #     pred = torch.argmax(iptresnet.tokenizer(patches), dim=1)

    #     adv_patches = adv_perturb(patches, iptresnet.tokenizer, pred, args.eps, args.attack_iters)
    #     adv_pred = torch.argmax(iptresnet.tokenizer(adv_patches), dim=1)

    #     correct += (adv_pred == pred).sum()
    #     total += pred.numel()
    # print(f'epoch: {epoch}| attacked iptresnet.tokenizer accuracy: {correct/total:.4f}')

    tok_train_set = tokenize_dataset(train_loader, iptresnet.tokenizer, args.patch_size, args.device)
    tok_train_loader = get_dataloader(tok_train_set, batch_size=args.batch_size)

    train_classifier(args, iptresnet, tok_train_loader, test_loader)