import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.cluster import KMeans

from tqdm import tqdm

from utils import *

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Define the architecture
        self.conv1 = CustomConv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = CustomConv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(32 * 7 * 7, num_classes)  # Adjust the size here depending on your input image size

    def forward(self, x):
        # Apply custom convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.fc(x)
        return x


class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(CustomConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
        # For storing patches and cluster centers
        self.max_patches = 1280
        self.patch_buffer = None
        self.patches = None
        self.num_centers = 128
        self.cluster_centers = None
        self.temperature = 100

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Extract patches
        this_patch = F.unfold(x, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        if self.patches is None:
            self.patches = this_patch.detach()
        else:
            self.patches = torch.cat((self.patches, this_patch.detach()), dim=0)

        if self.patches.size(0) > self.max_patches:
            indices = torch.randperm(self.patches.size(0))[:self.max_patches]
            self.patches = self.patches[indices, :, :]

        # self.patches = nn.Parameter(self.patch_buffer)
        self.cluster_centers = self.perform_clustering(self.patches, self.cluster_centers)

        # Transform patches to closest cluster centers
        transformed_patches = self.transform_to_clusters(this_patch, self.cluster_centers)

        new_size = x.size(3)*self.kernel_size[0]
        x_dash = F.fold(transformed_patches, output_size=(new_size, new_size), kernel_size=self.kernel_size, padding=self.padding, stride=self.kernel_size[0])

        output = F.conv2d(x_dash, self.weight, self.bias, self.kernel_size[0], self.padding, self.dilation, self.groups)
        return output

    def perform_clustering(self, patches, cluster_centers, num_iterations=10):
        # Reshape patches to (num_patches, flattened_patch_size)
        flat_patches = patches.view(-1, patches.size(1))

        # Initialize cluster centers randomly
        if cluster_centers is None:
            indices = torch.randperm(flat_patches.size(0))[:self.num_centers]
            cluster_centers = flat_patches[indices]

        for _ in range(num_iterations):
            # Compute distances from patches to cluster centers
            distances = torch.cdist(flat_patches, cluster_centers)

            # Assign each patch to the nearest cluster
            labels = torch.argmin(distances, dim=1)

            # Update cluster centers
            for i in range(self.num_centers):
                cluster_patches = flat_patches[labels == i]
                if len(cluster_patches) > 0:
                    cluster_centers[i] = cluster_patches.mean(dim=0)

        return cluster_centers

    def transform_to_clusters(self, patches, cluster_centers):
        # Reshape patches to (num_patches, flattened_patch_size)
        flat_patches = patches.view(-1, patches.size(1))
        # Calculate distances from each patch to each cluster center
        distances = torch.cdist(flat_patches, cluster_centers)
        # Find the index of the closest cluster center for each patch
        soft_assignments = F.softmax(-distances / self.temperature, dim=1)
        # Weighted sum of cluster centers based on the soft assignments
        transformed_patches = soft_assignments @ cluster_centers
        # Reshape back to the original patches' shape but with transformed values
        transformed_patches = transformed_patches.view_as(patches)
        return transformed_patches


if __name__ == '__main__':
    from data import get_dataloader
    model = SimpleCNN()
    trainloader, testloader = get_dataloader()
    device = torch.device("cuda:"+str(0))

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        pbar = tqdm(trainloader, ncols=88, desc='train')
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(trainloader)}')

    print('Training finished!')
    # torch.save(model.state_dict(), modelpath)

