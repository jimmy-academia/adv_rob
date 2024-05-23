import torch
import argparse
from networks import _simpleCNN
from data import get_dataloader
from experiments import run_bare, run_experiment
from pathlib import Path

from utils import *

def threelayer(args):
    # do bare baseline
    print('three layers!')
    Result = {}
    for i in range(2**3):
        layer_code = format(i, 'b').zfill(3)
        code = 'tl'+layer_code
        print(code)
        args.checkpoint_dir = Path('ckpt')/code
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(code=code)
        trainloader, testloader = get_dataloader(dataset=args.dataset, batch_size=args.batch_size)
        model = _simpleCNN(args, [16, 'r', 'p', 32, 'r', 'p', 64, 'r', 'f', 'l'], [char == '1' for char in layer_code])
        testacc, attackacc = run_experiment(args, model, trainloader, testloader)
        Result[code] = [testacc, attackacc]
    dumpj(Result, 'threelayer.json')


def main(args):
    # do bare baseline
    args.checkpoint_dir = Path('ckpt/default_main')
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(code='default_main')
    trainloader, testloader = get_dataloader(dataset=args.dataset, batch_size=args.batch_size)
    model = _simpleCNN(args, [16, 'r', 'p', 32, 'r', 'p', 64, 'r', 'f', 'l'], [False, False, True])
    # model = _simpleCNN(args)
    # normal_model = _simpleCNN(args, )
    print(model)

    run_experiment(args, model, trainloader, testloader)


def debug(args):
    args.checkpoint_dir = Path('ckpt/debug_three')
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(code='default_main')
    trainloader, testloader = get_dataloader(dataset=args.dataset, batch_size=args.batch_size)
    model = _simpleCNN(args, [16, 'r', 'p', 32, 'r', 'p', 64, 'r', 'f', 'l'], [False, False, False])
    # model = _simpleCNN(args)
    # normal_model = _simpleCNN(args, )
    print(model)
    print('check_final!!')
    print(run_experiment(args, model, trainloader, testloader))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiments on Custom CNN")
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for data loader')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--incr_epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')

    parser.add_argument('--epsilon', type=float, default=0.03)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--num_iter', type=int, default=20)

    parser.add_argument('--init_temp', type=float, default=1e3) # 1e-5
    parser.add_argument('--final_temp', type=float, default=1e10) # 1e5
    parser.add_argument('--num_centers', type=int, default=128)
    parser.add_argument('--num_cluster_iterations', type=int, default=3)
    parser.add_argument('--max_patch_count', type=int, default=5280)
    parser.add_argument('--patch_counts_gamma', type=float, default=1) #0.9

    parser.add_argument('--checkpoint_dir', type=str, default='ckpt/default')
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()
    args.device = torch.device("cpu" if args.device == -1 else "cuda:"+str(args.device))
    args.build_proto = True
    
    # main(args)
    # threelayer(args)
    debug(args)

## next: init temp, final temp 1e-5, 5; next: survey check() stage2 LARGER cluster_CENTER!! 