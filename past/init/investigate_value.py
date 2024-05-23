import torch
import torchvision
# import numpy as np
import torchvision.transforms as transforms

from utils import *
from sklearn.cluster import KMeans

Rootdir = '/home/jimmyyeh/Documents/CRATER/DATASET'
# Import MNIST dataset
trainset_mnist = torchvision.datasets.MNIST(root=Rootdir, train=True, download=True, transform=transforms.ToTensor())
testset_mnist = torchvision.datasets.MNIST(root=Rootdir, train=False, download=True, transform=transforms.ToTensor())

# Import CIFAR-10 dataset
trainset_cifar10 = torchvision.datasets.CIFAR10(root=Rootdir, train=True, download=True, transform=transforms.ToTensor())
testset_cifar10 = torchvision.datasets.CIFAR10(root=Rootdir, train=False, download=True, transform=transforms.ToTensor())


# Collect images from trainset_mnist and record pixel values
list_pixel_values = []
k = 2  # Specify the size of the sampling window


for i, (image, _) in enumerate(trainset_cifar10):
    h, w = image.shape[1:]
    for y in range(0, h, k):
        for x in range(0, w, k):
            window = image[:, y:y+k, x:x+k]
            window_vector = window.flatten()
            list_pixel_values.append(window_vector)
    if i == 99:
        break
pixel_values = torch.stack(list_pixel_values).numpy()

# Create a histogram of pixel values
# import matplotlib.pyplot as plt

# plt.hist(pixel_values.flatten(), bins=256, range=(0, 1))
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.title('Histogram of Pixel Values')
# plt.savefig('histogram.jpg')

# Perform K-means clustering
num_clusters = 2048  # Specify the number of clusters
kmeans = KMeans(n_clusters=num_clusters, n_init='auto', max_iter=300, random_state=0)
kmeans.fit(pixel_values)

cluster_labels = kmeans.predict(pixel_values)
#l2
distances = kmeans.transform(pixel_values)
assigned_cluster_distances = distances[range(len(distances)), cluster_labels]
# linfty
# distances = pixel_values - kmeans.cluster_centers_[cluster_labels]
# assigned_cluster_distances = abs(distances).max(axis=1)


for label in range(num_clusters):
    distances = assigned_cluster_distances[cluster_labels == label]
    distances = torch.Tensor(distances)
    max_deviation = torch.max(distances)
    average_deviation = torch.mean(distances)
    medium_deviation = torch.median(distances)
    print(f"Label {label}: Num {len(distances)}, Maximum Deviation = {max_deviation}, Medium Deviation = {medium_deviation}, Average Deviation = {average_deviation}")

# Plot histogram of distances from the cluster center
import matplotlib.pyplot as plt
plt.hist(assigned_cluster_distances.flatten(), bins=256, range=(0, 1))
plt.xlabel('Distance from Cluster Center')
plt.ylabel('Frequency')
plt.title('Histogram of Distances from Cluster Center')
plt.savefig('distance_histogram.jpg')
plt.close()
print('done')