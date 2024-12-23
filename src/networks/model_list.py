import torch.nn as nn
import torchvision.models as models

from networks.lenet import LeNet
from networks.iptnet import IPTNet
from networks.test_time import TTTBasicModel, ResNetCifar


class Dummy(nn.Module):
    def __init__(self, iptnet, classifier):
        super(Dummy, self).__init__()
        self.iptnet = iptnet
        self.classifier = classifier
    
    def forward(self, x):
        x = self.iptnet(x)
        x = self.classifier(x)
        return x

def dummy_model(args):
    classifier_type, ipt_type = args.model.split('_')
    model = Dummy(IPTNet(args), globals()[classifier_type](args))
    return model

def lenet(args):
    model = LeNet(args)
    return model

def efficientnet(args):
    model = models.efficientnet_v2_s()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, args.num_classes)
    return model

def mobilenet(args):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, args.num_classes)
    return model

def resnet4(args):
    model = ResNetCifar(depth=26, classes=args.num_classes)
    return model

def tttbasic(args):
    model = TTTBasicModel(args)
    return model