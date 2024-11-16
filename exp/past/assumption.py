import sys 
sys.path.append('.')
import torch

from utils import default_args
from datasets import get_dataloader
from main import post_process_args

from debug import *

def main():

    args = default_args()
    args = post_process_args(args)
    train_loader, test_loader = get_dataloader(args)

    patch_size = 2
    # eps = 8/255
    thresholds = [32, 16] + list(range(8, 0, -1))
    print(thresholds)
    for count, (images, __) in enumerate(train_loader):
        patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(-1, args.channels, patch_size, patch_size)
        mean = patches.mean(dim=[1,2,3], keepdim=True)
        diff = patches - mean

        t_list = []
        ratios = torch.arange(7) 
        for t in thresholds:
            under = abs(diff) > t/255
            under = under.sum([1,2,3])
            percentages = ((under.unsqueeze(1) <= ratios).sum(dim=0).float() / under.numel()) * 100
            t_list.append(percentages)
        t_list = torch.stack(t_list)

        torch.set_printoptions(precision=2)
        print(t_list.T.mean(0))
        if count == 10:
            break




    # make into patch data
    # check assumption

if __name__ == '__main__':
    main()