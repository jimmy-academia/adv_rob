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

from utils import default_args
from main import post_process_args
from datasets import get_datasets

from printer import display_images_in_grid

from debug import check


import matplotlib.pyplot as plt

def main():

    print('### calculate the lambda values for each order')

    args = default_args()
    args = post_process_args(args)
    args.vocab_size = 8
    args.patch_size = 2

    args.ckptdir = Path('ckpt')/'calc'
    args.ckptdir.mkdir(exist_ok=True)

    # === 

    train_set, test_set = get_datasets(args)

    print('train')
    examine_set(args, train_set, 'train')
    print('test')
    examine_set(args, test_set, 'test')
    

def examine_set(args, _set, name):
    
    lin_coeff = lambda x: [1 - 2 * (i / (x - 1)) for i in range(x)]
    list_wh = lin_coeff(args.patch_size)
    A = torch.tensor(list(product([1,0,-1], list_wh, list_wh)))
    num_patch = args.image_size//args.patch_size

    max_Collections = {'first_remain':[], 'first_order':[], 'second_remain':[]}
    avg_Collections = {'first_remain':[], 'first_order':[], 'second_remain':[]}

    count = 0
    for img, lab in tqdm(_set, ncols=80):
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

        # display_images_in_grid(args.ckptdir/'check.jpg', [x.unsqueeze(0) for x in [img, zeroth_order, first_remain, first_order, second_remain]]) 

        # collect infty norm: first remain, first order, second remain 
        max_Collections['first_remain'].append(first_remain.abs().max())
        max_Collections['first_order'].append(first_order.abs().max())
        max_Collections['second_remain'].append(second_remain.abs().max())

        avg_Collections['first_remain'].append(first_remain.abs().mean())
        avg_Collections['first_order'].append(first_order.abs().mean())
        avg_Collections['second_remain'].append(second_remain.abs().mean())

    plot_distributions(max_Collections, avg_Collections, args.ckptdir/f'{name}dist.jpg')
    print(name, 'distribution plotted!!')
    # plot the distribution of the max and avg norms

def plot_distributions(max_Collections, avg_Collections, imgpath):
    # Prepare data for plotting
    categories = ['first_remain', 'first_order', 'second_remain']
    
    # Plot max norms
    plt.figure(figsize=(12, 6))
    for i, category in enumerate(categories):
        plt.subplot(2, 3, i + 1)
        plt.hist(max_Collections[category], bins=20, color='skyblue', edgecolor='black')
        plt.title(f'Max Norm Distribution - {category}')
        plt.xlabel('Max Norm')
        plt.ylabel('Frequency')

    # Plot average norms
    for i, category in enumerate(categories):
        plt.subplot(2, 3, i + 4)
        plt.hist(avg_Collections[category], bins=20, color='salmon', edgecolor='black')
        plt.title(f'Avg Norm Distribution - {category}')
        plt.xlabel('Avg Norm')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(imgpath)


if __name__ == '__main__':
    main()


