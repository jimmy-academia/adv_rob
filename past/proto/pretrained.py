from utils import *

import torch
import torchvision
from torchvision.transforms import v2
from torchvision.models import resnet50, ResNet50_Weights

import torch.nn.functional as F

from tqdm import tqdm

def main():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    default_transform = torchvision.transforms.Compose([
            v2.Resize(224),
            v2.CenterCrop(224),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),  # to float32 in [0, 1]
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # typically from ImageNet
        ])
    imagenette_data = torchvision.datasets.Imagenette('../../DATASET/', split='val', transform=default_transform)
    #split = train, val
    data_loader = torch.utils.data.DataLoader(imagenette_data,
            batch_size=16, shuffle=True, num_workers=8)


    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    layer_name = 'layer1.0'
    layer = dict([*model.named_modules()])[layer_name]
    layer.register_forward_hook(get_activation(layer_name))


    device = 0
    model.to(device)

    # cluster_centers = None
    clusterer = DBSTREAM()


    total = correct = 0
    for images, labels in tqdm(data_loader, desc='data', ncols=88):
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        _, predicted = torch.max(output.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # feed_activations(activations[layer_name], cluster_centers)
        patches = F.unfold(activations[layer_name], kernel_size=2, dilation=1, padding=0, stride=2).permute(0, 2, 1).reshape(-1, 1024)
        clusterer.fit(patches)

    testacc = correct / total 
    print(testacc)

    for images, labels in tqdm(data_loader, desc='data2', ncols=88):
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)

        patches = F.unfold(activations[layer_name], kernel_size=2, dilation=1, padding=0, stride=2).permute(0, 2, 1).reshape(-1, 1024)

        btw_clust, to_clust = clusterer.show_distance(patches)

        check()
    

class DBSTREAM: 
    def __init__(self, radius=10, merge_threshold=30):
        self.radius = radius  # The radius to consider for the density calculation
        self.labels = []  # Store labels for each data point
        self.device = 0
        self.batch_size = 128
        self.clusters = torch.empty(0, device=self.device)  # Initialize an empty tensor for cluster centers on the specified device

    def fit(self, data):
        data = data.to(self.device)
        for i in range(0, data.size(0), self.batch_size):
            batch = data[i:i+self.batch_size]  # Get current batch
            if self.clusters.nelement() == 0:
                self.clusters = batch[:1]  # Start with the first point of the first batch as the first cluster if no clusters exist yet
            distant_M = torch.cdist(batch, self.clusters, 2)

            ### incorporate all batch data with distant_M > self.radius for all current clusters as new cluster centers
            mask = torch.all(distant_M > self.radius, dim=1)
            new_centers = batch[mask]
            if new_centers.nelement() > 0:
                self.clusters = torch.cat((self.clusters, new_centers), dim=0)

            
    def predict(self, data):
        # data is expected to be a PyTorch tensor of shape (n_samples, n_features)
        labels = torch.empty(data.size(0), dtype=torch.long)
        for i, point in enumerate(data):
            distances = torch.stack([torch.norm(cluster - point) for cluster in self.clusters])
            label = torch.argmin(distances)
            labels[i] = label
        return labels


# check()
                # for point in batch:
                #     point = point.unsqueeze(0)  # Add batch dimension for broadcasting
                #     distances = torch.norm(self.clusters - point, dim=1)
                #     if not torch.any(distances < self.radius):
                #         self.clusters = torch.cat((self.clusters, point), dim=0)


# labels = model.predict(data)
# print(labels)


# def feed_activations(act_tensor, cluster_centers):
#     # shape 16, 256, 56, 56
#     patches = F.unfold(act_tensor, kernel_size=2, dilation=1, padding=0, stride=2).permute(0, 2, 1).reshape(-1, 1024)

#     if cluster_centers is None:
#         return 



if __name__ == '__main__':
    main()