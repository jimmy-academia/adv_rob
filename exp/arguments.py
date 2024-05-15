import torch
from types import SimpleNamespace
from pathlib import Path

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
