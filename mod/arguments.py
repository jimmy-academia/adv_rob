import torch
from types import SimpleNamespace
from pathlib import Path
import argparse

def set_arguments():
    parser = argparse.ArgumentParser(
        description='Tokenizer and Classifier Training with Adversarial Evaluation', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # general
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str, default='0', help='gpu device id')
    # dataset
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=1024, help='dataset batch size')
    # tokenizer
    parser.add_argument('--patch_size', type=int, default=8, help='patch size to be tokenized')
    parser.add_argument('--vocabulary_size', type=int, default=1024, help='vocabulary size of token')
    parser.add_argument('--num_hidden_layer', type=int, default=2, help='number of hidden layers in tokenizer')
    # training and attack
    parser.add_argument('--precluster_method', type=str, default='kmeans', choices=['None', 'kmeans'], help='clustering method for token embedding')
    parser.add_argument('--toktrain_epochs', type=int, default=10, help='tokenizer training epochs')
    # parser.add_argument('--tok_batch_iters', type=int, default=1024, help='tokenizer training batch iters')
    parser.add_argument('--train_epochs', type=int, default=128, help='classifier training epochs')
    parser.add_argument('--attack_iters', type=int, default=100, help='classifier attack iterations')
    # ckpt
    parser.add_argument('--ckpt_dir', type=str, default='ckpt', help='checkpoint root directory')
    args = parser.parse_args()

    ## post process args
    args.device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device('cpu')
    ## dataset defined
    args.input_channels = 1 if args.dataset == 'mnist' else 3
    args.image_size = 224 if args.dataset == 'imagenet' else 32
    args.embed_size = args.patch_size**2
    args.eps = 0.3 if args.dataset == 'mnist' else 0.03
    num_class_dict = {'mnist': 10, 'cifar10': 10, 'cifar100':100, 'imagenet': 1000}
    args.num_classes = num_class_dict[args.dataset]
    ## paths
    args.ckpt_dir = Path(args.ckpt_dir)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.tokenizer_path = args.ckpt_dir/f'tokenizer_{args.dataset}_{args.patch_size}_{args.vocabulary_size}.pth'
    args.classifier_path = args.ckpt_dir/'classifier.pth'
    return args

