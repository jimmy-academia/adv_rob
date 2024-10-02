from networks.mcunet import *
from networks.iptnet import * 
from networks.predefined import * 

def get_model(args):
    input_size = 32 if args.dataset != 'imagenet' else 256
    num_classes = 10 if args.dataset not in ['imagenet', 'cifar100'] else (100 if args.dataset == 'cifar100' else 1000)
    model = globals()[args.model](nc=num_classes, in_sz = input_size)

    return model