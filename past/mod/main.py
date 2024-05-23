
from parse import set_arguments
from utils import set_seeds
from data import get_dataset, get_dataloader
from networks import IPTResnet
from train import train

def main():
    args = set_arguments()
    set_seeds(args.seed)
    train_set, test_set = get_dataset(args.dataset)
    train_loader, test_loader = get_dataloader(train_set, test_set, args.batch_size)

    iptresnet = IPTResnet(args).to(args.device)
    train(args, iptresnet, train_loader, test_loader)
    

if __name__ == '__main__':
    main()