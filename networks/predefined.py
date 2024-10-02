import torch.nn as nn
import torchvision.models as models

def resnet(depth, **kwargs):
    model_map = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        # Add more if needed
    }
    return model_map[depth](**kwargs)

def mobilenet(in_sz, nc):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, nc)
    return model