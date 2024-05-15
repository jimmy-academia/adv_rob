import torch
from types import SimpleNamespace
from pathlib import Path
import argparse

def set_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--dataset', type=str, default='mnist')
    
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--vocabulary_size', type=int, default=1024)
    parser.add_argument('--num_hidden_layer', type=int, default=2)
    parser.add_argument('--embed_size', type=int, default=64)

    parser.add_argument('--toktrain_epochs', type=int, default=1024)
    parser.add_argument('--tok_batch_iters', type=int, default=1024)
    parser.add_argument('--train_epochs', type=int, default=128)
    parser.add_argument('--attack_iters', type=int, default=100)

    parser.add_argument('--ckpt_dir', type=str, default='ckpt')
    args = parser.parse_args()

    ## post process args
    args.device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device('cpu')
    ## dataset defined
    args.image_size = 224 if args.dataset == 'imagenet' else 32
    args.eps = 0.3 if args.dataset == 'mnist' else 0.03
    num_class_dict = {'mnist': 10, 'cifar10': 10, 'cifar100':100, 'imagenet': 1000}
    args.num_classes = num_class_dict[args.dataset]
    ## paths
    args.ckpt_dir = Path(args.ckpt_dir)
    args.tokenizer_path = args.ckpt_dir/f'tokenizer_{args.dataset}_{args.patch_size}_{args.vocabulary_size}.pth'
    args.classifier_path = args.ckpt_dir/'classifier.pth'
    return args

def default_args():
    args = SimpleNamespace()
    args.seed = 0
    args.device = torch.device("cuda:0")
    args.dataset = 'mnist'

    args.patch_size = 8
    args.vocabulary_size = 1024
    args.image_size = 224 if args.dataset == 'imagenet' else 32
    args.eps = 0.3 if args.dataset == 'mnist' else 0.03
    args.num_hidden_layer = 2
    args.embed_size = args.patch_size**2
    args.num_classes = 10

    args.toktrain_epochs = 1024
    args.tok_batch_iters = 1024
    args.train_epochs = 128
    args.attack_iters = 100

    # paths
    args.ckpt_dir = Path('ckpt')
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.tokenizer_path = args.ckpt_dir/f'tokenizer_{args.dataset}_{args.patch_size}_{args.vocabulary_size}.pth'
    args.classifier_path = args.ckpt_dir/'classifier.pth'
    return args
