from utils import *

import torch
import torchvision
from torchvision.transforms import v2
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
# print(model)

# imagenet_data = torchvision.datasets.ImageNet('../../DATASET/')
default_transform = torchvision.transforms.Compose([
        v2.Resize(224),
        v2.CenterCrop(224),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),  # to float32 in [0, 1]
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # typically from ImageNet
    ])
imagenette_data = torchvision.datasets.Imagenette('../../DATASET/', transform=default_transform)
#split = train, val
data_loader = torch.utils.data.DataLoader(imagenette_data,
                                          batch_size=10,
                                          shuffle=True,
                                          num_workers=8, 
                                          )

for images, labels in data_loader:
    check()
    output = model(images)

activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

layer_name = 'layer1.0'
layer = dict([*model.named_modules()])[layer_name]
layer.register_forward_hook(get_activation(layer_name))


# hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")

# # from datasets import load_dataset

# # If the dataset is gated/private, make sure you have run huggingface-cli login
# # dataset = load_dataset("imagenet-1k")


# from torch.utils.data import DataLoader

# def collate_fn(examples):
#     images = []
#     labels = []
#     for example in examples:
#         images.append((example["pixel_values"]))
#         labels.append(example["labels"])
        
#     pixel_values = torch.stack(images)
#     labels = torch.tensor(labels)
#     return {"pixel_values": pixel_values, "labels": labels}
# dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=4)