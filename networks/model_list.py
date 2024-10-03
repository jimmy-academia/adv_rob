import torch.nn as nn
import torchvision.models as models

from networks.iptnet import APTNet
from networks.test_time import TTTBasicModel


class Dummy(nn.Module):
    def __init__(self, iptnet, classifier):
        super(Dummy, self).__init__()
        self.iptnet = iptnet
        self.classifier = classifier
    
    def forward(self, x):
        x = self.iptnet(x)
        x = self.classifier(x)
        return x

def mobilenet(args):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, args.num_classes)
    return model

def mobileapt(args):
    model = Dummy(APTNet(args), mobilenet(args))
    return model

def tttbasic(args):
    model = TTTBasicModel(args)
    return model