import torch
import argparse
from networks import SimpleCNN
from data import get_dataloader
from experiments import run_experiment
from utils import *
from pathlib import Path

def main(args):
    # Setup logging and other configurations
    
    trainloader, testloader = get_dataloader(dataset=args.dataset, batch_size=args.batch_size)

    # Grid search over final_temp and num_centers
    for temp_pow in range(2, 7):
        for cen_pow in range(3, 8):
            args.final_temp = 10**temp_pow
            args.num_centers = 2**cen_pow
            
            code = f't{temp_pow}c{cen_pow}'
            setup_logging(code=code)

            # Unique checkpoint directory for each combination
            args.checkpoint_dir = Path(f'ckpt/{code}')

            print(f"Running experiment with final_temp: {args.final_temp}, num_centers: {args.num_centers}")
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

    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()
    args.device = torch.device("cpu" if args.device == -1 else "cuda:"+str(args.device))

    main(args)

# if __name__ == '__main__':
#     from utils import *
#     from networks import SimpleCNN
#     model = SimpleCNN()


# nmod = model.create_equivalent_normal_cnn()
# a = torch.rand(3,1,28,28)
# model(a)
# nmod(a)
