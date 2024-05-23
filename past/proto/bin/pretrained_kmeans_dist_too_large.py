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
    clusterer = MiniBatchKMeans()


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
    


class MiniBatchKMeans:
    def __init__(self, n_clusters=128, max_iter=10, batch_size=100, tol=1e-4, device='cuda'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.device = device
        self.centroids = None

        self.dist_btw_clusters = None

    def _init_centroids(self, data):
        """Initialize centroids by randomly selecting points from the dataset."""
        init_indices = torch.randperm(data.size(0))[:self.n_clusters]
        self.centroids = data[init_indices].to(self.device)

    def _assign_clusters(self, data):
        """Assign data points to the nearest centroid."""
        distances = torch.cdist(data, self.centroids)
        return torch.argmin(distances, dim=1)

    # def _update_centroids(self, data, labels):
    #     """Update centroids by computing the mean of assigned points."""
    #     new_centroids = []
    #     for i in range(self.n_clusters):
    #         cluster_points = data[labels == i]
    #         if len(cluster_points) > 0:
    #             new_centroids.append(cluster_points.mean(dim=0))
    #         else:  # If a centroid loses all its points, reinitialize it
    #             new_centroids.append(self.centroids[i])
    #     self.centroids = torch.stack(new_centroids)

    def _update_centroids(self, data, labels):
        # Create an empty tensor for new centroids
        new_centroids = torch.zeros(self.n_clusters, data.size(1), device=self.device)
        
        # Count the number of points assigned to each centroid
        counts = torch.zeros(self.n_clusters, device=self.device).index_add_(0, labels, torch.ones_like(labels, dtype=torch.float))
        
        # Sum data points assigned to each centroid
        sum_points = new_centroids.scatter_add_(0, labels.unsqueeze(1).expand(-1, data.size(1)), data)
        
        # Calculate the mean for centroids with at least one assigned point
        nonzero_counts = counts > 0
        new_centroids[nonzero_counts] = sum_points[nonzero_counts] / counts[nonzero_counts].unsqueeze(1)
        
        # Handle centroids with no points assigned (keeping the old centroids in such cases)
        new_centroids[~nonzero_counts] = self.centroids[~nonzero_counts]
        
        self.centroids = new_centroids


    def fit(self, data):
        """Fit the model to the data using Mini-Batch K-Means."""
        data = data.to(self.device)
        if self.centroids is None:
            self._init_centroids(data)

        for _ in range(self.max_iter):
            centroid_shift = 0
            for i in range(0, data.size(0), self.batch_size):
                batch = data[i:i+self.batch_size]
                labels = self._assign_clusters(batch)
                prev_centroids = self.centroids.clone()
                self._update_centroids(batch, labels)
                centroid_shift += torch.norm(self.centroids - prev_centroids)

            if centroid_shift < self.tol:
                break

    def predict(self, data):
        """Assign clusters to data points."""
        data = data.to(self.device)
        labels = self._assign_clusters(data)
        return labels.cpu()

    def show_distance(self, data):
        ## distance to cluster vs distance between clusters
        data = data.to(self.device)
        labels = self._assign_clusters(data)

        # calculate distance between clusters if not calculated before
        if self.dist_btw_clusters is None:
            cluster_distances = torch.cdist(self.centroids, self.centroids, p=2)
            # Since the distance of a cluster to itself is 0, we mask those values to ignore them
            cluster_distances.masked_fill_(cluster_distances == 0, float('inf'))
            min_dist_btw_clusters = torch.min(cluster_distances)
            self.dist_btw_clusters = min_dist_btw_clusters

        # calculate average distance from ptach to assigned clusters 
        # Calculate average distance from data points to assigned clusters
        distances = torch.cdist(data, self.centroids, p=2)
        # For each data point, select the distance to its assigned cluster
        dist_to_assigned_cluster = distances.gather(1, labels.unsqueeze(1)).squeeze()
        avg_dist_to_cluster = torch.mean(dist_to_assigned_cluster)

        return self.dist_btw_clusters, avg_dist_to_cluster


### not finished...
# class DBSTREAMBasic: 
#     def __init__(self, radius=10, min_samples=5, merge_threshold=30):
#         self.radius = radius  # The radius to consider for the density calculation
#         self.min_samples = min_samples  # Minimum number of samples in a neighborhood to form a cluster
#         self.labels = []  # Store labels for each data point
#         self.device = 0
#         self.batch_size = 128
#         self.clusters = torch.empty(0, device=self.device)  # Initialize an empty tensor for cluster centers on the specified device

#     def fit(self, data):
#         data = data.to(self.device)
#         for i in range(0, data.size(0), self.batch_size):
#             batch = data[i:i+self.batch_size]  # Get current batch
#             if self.clusters.nelement() == 0:
#                 self.clusters = batch[:1]  # Start with the first point of the first batch as the first cluster if no clusters exist yet
#             torch.cdist(batch, self.clusters, 2)
#             distant_M = torch.norm(batch.unsqueeze(1) - self.clusters.unsqueeze(0), 2)

#                 for point in batch:
#                     point = point.unsqueeze(0)  # Add batch dimension for broadcasting
#                     distances = torch.norm(self.clusters - point, dim=1)
#                     if not torch.any(distances < self.radius):
#                         self.clusters = torch.cat((self.clusters, point), dim=0)


#     def predict(self, data):
#         # data is expected to be a PyTorch tensor of shape (n_samples, n_features)
#         labels = torch.empty(data.size(0), dtype=torch.long)
#         for i, point in enumerate(data):
#             distances = torch.stack([torch.norm(cluster - point) for cluster in self.clusters])
#             label = torch.argmin(distances)
#             labels[i] = label
#         return labels

# labels = model.predict(data)
# print(labels)


# def feed_activations(act_tensor, cluster_centers):
#     # shape 16, 256, 56, 56
#     patches = F.unfold(act_tensor, kernel_size=2, dilation=1, padding=0, stride=2).permute(0, 2, 1).reshape(-1, 1024)

#     if cluster_centers is None:
#         return 



if __name__ == '__main__':
    main()