import torch
import argparse
from networks import SimpleCNN
from data import get_dataloader
from experiments import run_bare, run_experiment
from pathlib import Path

from utils import *


def bare(args):
    # do bare baseline
    args.checkpoint_dir = Path('ckpt/default_bare')
    setup_logging(code='default_bare')
    trainloader, testloader = get_dataloader(dataset=args.dataset, batch_size=args.batch_size)
    model = SimpleCNN(args)
    run_bare(args, model, trainloader, testloader)


def main(args):
    # do bare baseline
    args.checkpoint_dir = Path('ckpt/default_main')
    setup_logging(code='default_main')
    trainloader, testloader = get_dataloader(dataset=args.dataset, batch_size=args.batch_size)
    model = SimpleCNN(args)
    run_experiment(args, model, trainloader, testloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiments on Custom CNN")
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for data loader')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')

    parser.add_argument('--epsilon', type=float, default=0.03)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--num_iter', type=int, default=20)


    parser.add_argument('--init_temp', type=float, default=1e-5) #100000
    parser.add_argument('--final_temp', type=float, default=1e5) #100000
    parser.add_argument('--num_centers', type=int, default=256)

    parser.add_argument('--checkpoint_dir', type=str, default='ckpt/default')
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()
    args.device = torch.device("cpu" if args.device == -1 else "cuda:"+str(args.device))

    bare(args)
