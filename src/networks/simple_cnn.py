import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SimpleCNN(nn.Module):
    def __init__(self, args):
        super(SimpleCNN, self).__init__()
        num_classes = 10 if args.dataset == 'mnist' else None

        self.conv1 = CustomConv2d(args, in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = CustomConv2d(args, in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(32 * 7 * 7, num_classes)
        self.temp = 1e-6

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def set_mode(self, mode):
        if mode not in ['train', 'eval']:
            raise ValueError("mode should be 'train' or 'eval'")
        for layer in [self.conv1, self.conv2]:
            layer.set_mode(mode)

    def set_temp(self, temp=None):
        if temp is not None:
            self.temp = temp
        for layer in [self.conv1, self.conv2]:
            layer.temp = self.temp

    def fetch_info(self):
        return [layer.fetch_info() for layer in [self.conv1, self.conv2]]

    def create_equivalent_normal_cnn(self):

        layers = []

        for module in self.children():
            if isinstance(module, CustomConv2d):
                conv_layer = nn.Conv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=module.bias is not None
                )
                layers.append(conv_layer)
            else:
                layers.append(module)

        normal_cnn = nn.Sequential(*layers)

        for simple_layer, normal_layer in zip(self.children(), normal_cnn.children()):
            if isinstance(normal_layer, nn.Conv2d):
                normal_layer.weight.data = simple_layer.weight.clone()
                if simple_layer.bias is not None:
                    normal_layer.bias.data = simple_layer.bias.clone()
            elif isinstance(normal_layer, nn.Linear):
                normal_layer.load_state_dict(simple_layer.state_dict())

        return normal_cnn

    def save_param(self, checkpoint_dir='ckpt', epoch=None, filename='model.pth'):
        """
        Saves the model's state_dict and additional info in one file. 
        Allows customization of the save filename.
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        filename_suffix = f".{epoch}" if epoch is not None else ""
        save_path = checkpoint_path / (filename + filename_suffix)
        save_data = {
            'state_dict': self.state_dict(),
            'info': self.fetch_info()
        }

        torch.save(save_data, save_path)

    def load_param(self, checkpoint_dir='ckpt', epoch=None, filename='model.pth'):
        """
        Loads the model's state_dict and additional info from a single file.
        """
        filename_suffix = f".{epoch}" if epoch is not None else ""
        load_path = Path(checkpoint_dir) / (filename + filename_suffix)
        data = torch.load(load_path)
        self.load_state_dict(data['state_dict'])

        info = data['info']
        self.conv1.patches, self.conv1.cluster_centers, _ = info[0]
        self.conv2.patches, self.conv2.cluster_centers, _ = info[1]
        self.temp = info[0][-1]
        self.set_temp()


class CustomConv2d(nn.Module):
    def __init__(self, args, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
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
        self.num_centers = args.num_centers

        self.patch_buffer = None
        self.patches = None
        self.cluster_centers = None
        self.is_eval=False
        self.temp = 1e-5


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Extract patches
        this_patch = F.unfold(x, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        
        # self.patches = nn.Parameter(self.patch_buffer)
        if not self.is_eval:
            if self.patches is None:
                self.patches = this_patch.detach()
            else:
                self.patches = torch.cat((self.patches, this_patch.detach()), dim=0)

            if self.patches.size(0) > self.max_patches:
                indices = torch.randperm(self.patches.size(0))[:self.max_patches]
                self.patches = self.patches[indices, :, :]

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
        soft_assignments = F.softmax(-distances * self.temp, dim=1)
        # Weighted sum of cluster centers based on the soft assignments
        transformed_patches = soft_assignments @ cluster_centers
        final_patches = self.temp/(self.temp + 1) * transformed_patches + 1/(self.temp + 1)*flat_patches
        # Reshape back to the original patches' shape but with transformed values
        final_patches = final_patches.view_as(patches)
        return final_patches

    def set_mode(self, mode):
        self.is_eval = mode == 'eval'

    def fetch_info(self):
        return [self.patches, self.cluster_centers, self.temp]
