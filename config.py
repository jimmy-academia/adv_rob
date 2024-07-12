import torch
from pathlib import Path
import argparse


def config_arguments(config):
    parser = argparse.ArgumentParser(description='IPT')
    # ony adjust general arguments with parser, others with yaml config file
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str, default='0', help='gpu device id')
    args = parser.parse_args()

    ## post process args
    args.device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device('cpu')

    # check common fields in yaml; individual fields checked in each experiment scripts
    for field in ['name', 'dataset', 'batch_size', 'patch_size', 'vocab_size']:
        assert field in config, f'{field} not defined in config file'
    
    args.config = config # save the whole config for detail configurations of model architecture or training procedure
    args.dataset = config['dataset']
    args.batch_size = config['batch_size']
    args.patch_size = config['patch_size']
    args.vocab_size = config['vocab_size']

    ## with dataset defined
    args.channels = 1 if args.dataset == 'mnist' else 3
    args.image_size = 224 if args.dataset == 'imagenet' else 32
    if 'conv' in config['model']['patcher_type']:
        args.patch_numel = args.channels * args.patch_size**2 
        args.num_patches_width = args.image_size // args.patch_size
    else:
        args.patch_numel = args.patch_size

    args.eps = 0.3 if args.dataset == 'mnist' else 8/255
    args.attack_iters = 100
    num_class_dict = {'mnist': 10, 'cifar10': 10, 'cifar100':100, 'imagenet': 1000}
    args.num_classes = num_class_dict[args.dataset]
    ## paths
    args.ckpt_dir = Path('ckpt')/config['name'] # mkdir prior to save; not here.
    # if args.ckpt_dir.exists():
    #     input(f'{args.ckpt_dir} already exists. Press any key to continue')
    # else:
    #     args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    return args


def default_arguments(dataset):
    args = argparse.Namespace(batch_size=128, dataset=dataset, seed=0)
    args.device = torch.device("cuda:0")
    args.channels = 1 if args.dataset == 'mnist' else 3
    args.image_size = 224 if args.dataset == 'imagenet' else 32
        
    args.patch_size = 2
    args.patch_numel = args.channels * args.patch_size**2 
    args.num_patches_width = args.image_size // args.patch_size

    args.vocab_size = 12
    args.eps = 0.3 if args.dataset == 'mnist' else 8/255
    num_class_dict = {'mnist': 10, 'cifar10': 10, 'cifar100':100, 'imagenet': 1000}
    args.num_classes = num_class_dict[args.dataset]
    args.attack_iters = 100
    return args


    