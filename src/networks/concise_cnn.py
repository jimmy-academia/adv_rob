'''
a more concise code that can be easily extendable

'''
from itertools import cycle

import torch.nn as nn
from .proto import ProtoConv2d

def _simpleCNN(args, layers=[16, 'r', 'p', 32, 'r', 'p', 'f', 'l'], proto_list=None):
    if args.dataset == 'mnist':
        args.input_size = 28
        args.classes = 10
    model = ConciseCNN(args, layers, proto_list)
    return model

class ConciseCNN(nn.Module):
    def __init__(self, args, layers, proto_list):
        super(ConciseCNN, self).__init__()
        self.args = args
        self.channels = 1 if args.dataset in ['mnist'] else 3

        self._end_size = args.input_size // (2 ** layers.count('p'))
        self.proto_iter = None if proto_list is None else iter(proto_list)
        # iter([False, True])

        modules = [self._make_layer(l) for l in layers]
        self.sequential = nn.Sequential(*modules)
        self.train = True
        self.temp = None

    def forward(self, x):
        for module in self.sequential:
            # Check if the module is an instance of ProtoConv2d
            if isinstance(module, ProtoConv2d):
                x = module(x, self.train, self.temp)
            else:
                x = module(x)
        return x

    def _make_layer(self, l):
        # Define your method to create a layer
        if type(l) is int:
            if self.proto_iter is None:
                if self.args.build_proto:
                    conv_layer = ProtoConv2d(self.args, self.channels, l, 3,1,1)
                else:
                    conv_layer = nn.Conv2d(self.channels, l, 3,1,1)
            else:
                if next(self.proto_iter):
                    conv_layer = ProtoConv2d(self.args, self.channels, l, 3,1,1)
                else:
                    conv_layer = nn.Conv2d(self.channels, l, 3,1,1)


            self.channels = l
            return conv_layer
        else:
            match l:
                case 'r':
                    return nn.ReLU()
                case 'p':
                    return nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                case 'f':
                    return nn.Flatten()
                case 'l':
                    return nn.Linear(self.channels* self._end_size**2, self.args.classes)


if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from utils import *
    from types import SimpleNamespace
    # from proto import ProtoConv2d

    args = SimpleNamespace()
    args.dataset = 'mnist'
    args.input_size = 28
    args.classes = 10
    args.build_proto = True

    layers = [16, 'r', 'p', 32, 'r', 'p', 'f', 'l']
    model = _simpleCNN(args, layers)

    import torch

    a = torch.rand(5, 1, 28, 28)
    
    check()