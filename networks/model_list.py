import torch.nn as nn
import torchvision.models as models

from networks.iptnet import APTNet
from networks.afanet import AFANet
from networks.zlqhnet import ZLQHNet
from networks.test_time import TTTBasicModel, ResNetCifar


IPT_Dict = {
    'apt': APTNet,
    'afa': AFANet,
    'zlqh': ZLQHNet,
}

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


def tttbasic(args):
    model = TTTBasicModel(args)
    return model

def resnetcifar(args):
    model = ResNetCifar(depth=26, classes=args.num_classes)
    return model

def dummy_models(args):
    classifier_type, ipt_type = args.model.split('_')
    model = Dummy(IPT_Dict[ipt_type](args), globals()[classifier_type](args))
    return model

# def mobilenet_apt(args):
#     model = Dummy(APTNet(args), mobilenet(args))
#     return model

# def resnetcifar_apt(args):
#     model = Dummy(APTNet(args), ResNetCifar(depth=26, classes=args.num_classes))
#     return model
    
# def resnetcifar_afa(args):
#     model = Dummy(AFANet(args), ResNetCifar(depth=26, classes=args.num_classes))
#     return model