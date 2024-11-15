'''
check the recon situation
'''
import sys 
sys.path.append('.')
import random
from itertools import product
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from utils import default_args
from main import post_process_args
from datasets import get_datasets

from printer import display_images_in_grid

from debug import check
from networks import get_model

import matplotlib.pyplot as plt

def main():

    print('### calculate if 2nd order tokenization is useful')
    print('### how well can tokenization help fit second_remain?')

    args = default_args()
    args = post_process_args(args)
    args.vocab_size = 8
    args.patch_size = 2
    args.model = 'resnetcifar_zlqh'

    args.ckptdir = Path('ckpt')/'calc'
    args.ckptdir.mkdir(exist_ok=True)

    # === 

    train_set, test_set = get_datasets(args)

    print('train')
    train_set = examine_set(args, train_set, 'train')
    test_set = examine_set(args, test_set, 'test')
    model = get_model(args)
    model = model.iptnet

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    train_model(args, model, train_loader, test_loader)
    
def examine_set(args, _set, name):
    
    lin_coeff = lambda x: [1 - 2 * (i / (x - 1)) for i in range(x)]
    list_wh = lin_coeff(args.patch_size)
    A = torch.tensor(list(product([1,0,-1], list_wh, list_wh)))
    num_patch = args.image_size//args.patch_size

    max_Collections = {'first_remain':[], 'first_order':[], 'second_remain':[]}
    avg_Collections = {'first_remain':[], 'first_order':[], 'second_remain':[]}

    count = 0
    remain_set = []
    for img, lab in tqdm(_set, ncols=80, desc='examining'):
        AvgPool = torch.nn.AvgPool2d(2)

        zeroth_order = AvgPool(img).repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        first_remain = img - zeroth_order 
        patches = first_remain.unfold(1, args.patch_size, args.patch_size).unfold(2, args.patch_size, args.patch_size)
        patches = patches.contiguous().view(3, -1, args.patch_size * args.patch_size)
        # first order
        
        patches = first_remain.unfold(1, args.patch_size, args.patch_size).unfold(2, args.patch_size, args.patch_size)
        patch_square = args.patch_size * args.patch_size
        patches = patches.contiguous().view(3, -1, patch_square)
        patches = patches.permute(1,0,2).contiguous().view(-1, 3*patch_square)
        A_expanded = A.unsqueeze(0).repeat(patches.size(0), 1, 1)  
        abc_list = torch.linalg.lstsq(A_expanded, patches)[0]

        first_order = torch.bmm(A_expanded, abc_list.unsqueeze(2))

        ref = A @ abc_list[1]
        first_order[1].reshape(-1) == ref
        first_order = first_order.view(-1, 3, args.patch_size, args.patch_size)
        first_order = first_order.view(num_patch, num_patch, 3, args.patch_size, args.patch_size)
        first_order = first_order.permute(2, 0, 3, 1, 4).contiguous().view(3, args.image_size, args.image_size)

        second_remain = first_remain - first_order
        remain_set.append(second_remain)

    return remain_set

def model_fit(model, x):
    x = model.high_predictor(x)     # bs, T, 16, 16
    x = x.permute(0, 2,3,1)     # bs, 16, 16, T
    x = x.view(x.size(0), -1, x.size(-1))   # bs, 256, T
    x = model.softmax(x)
    x = torch.matmul(x, model.embedding.weight)  # bs, 256, 12
    x = model.inverse(x)         # bs, 3, 32, 32
    return x

def train_model(args, model, train_loader, test_loader):
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters())

    pbar = tqdm(train_loader, ncols=88, desc='remain')
    for epoch in range(50):
        
        model.train()
        for remain_imgs in pbar:
            remain_imgs = remain_imgs.to(args.device)
            output = model_fit(model, remain_imgs)
            loss = torch.nn.MSELoss()(output, remain_imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(l=float(loss.cpu().detach()))

        if epoch == 0:
            display_images_in_grid(args.ckptdir/'remain_train.jpg', [remain_imgs[:5].cpu(), output[:5].cpu(), remain_imgs[:5].cpu() - output[:5].cpu()])
            

        model.eval()
        total_remain = 0
        for remain_imgs in test_loader:
            remain_imgs = remain_imgs.to(args.device)
            output = model_fit(model, remain_imgs)
            final_remain = remain_imgs - output
            total_remain += float(final_remain.abs().sum().cpu())

        if epoch == 0:
            display_images_in_grid(args.ckptdir/'remain_test.jpg', [remain_imgs[:5].cpu(), output[:5].cpu(), remain_imgs[:5].cpu() - output[:5].cpu()])


        print('epoch', epoch, 'train loss: ', float(loss.cpu().detach()), 'test_remain', total_remain)

if __name__ == '__main__':
    main()


