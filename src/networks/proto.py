import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *


class ProtoConv2d(nn.Module):
    def __init__(self, args, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ProtoConv2d, self).__init__()
        self.args = args
        self.patch_counts = [0]*args.num_centers

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups,kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # self.patch_size = 14 # 
        self.patch_size = self.kernel_size**2 * self.in_channels
        self.cluster_centers = torch.rand(args.num_centers, self.patch_size).to(args.device)
        

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, train=False, temp=None):
        if temp is None:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            # prototypical inference
            this_patch = F.unfold(x, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        
            # a = this_patch.view(-1, patch_size)
            # b = this_patch.transpose(1,2).reshape(-1, patch_size)

            flat_patches = this_patch.view(-1, self.patch_size)
            # flat_patches = this_patch.transpose(1,2).reshape(-1, patch_size)
            # flat_patches.reshape_as(this_patch.transpose(1,2)).transpose(1,2) == this_patch

            if train:
                # stream update center with this_patch 
                for _ in range(self.args.num_cluster_iterations):
                    # Compute distances from patches to cluster centers
                    distances = torch.cdist(flat_patches, self.cluster_centers)
                    labels = torch.argmin(distances, dim=1)
                    mask = torch.ones_like(distances, dtype=bool)
                    mask[torch.arange(distances.size(0)).unsqueeze(1), labels.unsqueeze(1)] = False
                    distances[mask].view(distances.size(0), -1)
                    weight = distances[mask].view(distances.size(0), -1).mean(1) - distances[torch.arange(distances.size(0)), labels] 
                    # Update cluster centers
                    ## contrastive sampling
                    for i in range(self.args.num_centers):
                        cluster_patches = flat_patches[labels == i].detach()
                        weight_i = weight[labels == i].detach()
                        if len(cluster_patches) > 0:
                            self.cluster_centers[i] = (self.cluster_centers[i] * self.patch_counts[i] + (cluster_patches * weight_i.unsqueeze(1)).sum(0)) / (self.patch_counts[i] + weight_i.sum())
                            self.patch_counts[i] += weight_i.sum()
                
                    # Update count i.e. weight/inertia of current cluster center
                    self.patch_counts[i] = min(self.patch_counts[i], self.args.max_patch_count)
                    self.patch_counts[i] *= self.args.patch_counts_gamma

            # Transform patches to closest cluster centers
            distances = torch.cdist(flat_patches, self.cluster_centers)
            # Find the index of the closest cluster center for each patch
            soft_assignments = F.softmax(-distances * temp, dim=1)
            # Weighted sum of cluster centers based on the soft assignments
            transformed_patches = soft_assignments @ self.cluster_centers
            final_patches = (temp * transformed_patches + flat_patches)/(temp + 1)
        
            # Reshape back to the original patches' shape but with transformed values
            final_patches = final_patches.view_as(this_patch)
            # final_patches = final_patches.reshape_as(this_patch.transpose(1,2)).transpose(1,2)

            new_size = x.size(3)*self.kernel_size  # calculate!!
            x_refold = F.fold(final_patches, output_size=(new_size, new_size), kernel_size=(self.kernel_size, self.kernel_size), padding=self.padding, stride=self.kernel_size)

            out = F.conv2d(x_refold, self.weight, self.bias, self.kernel_size, self.padding, self.groups)
            
            return out


if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from utils import *
    from types import SimpleNamespace

    temp = 10
    config = SimpleNamespace()
    config.mode = 'eval'
    config.num_centers = 128
    config.num_iterations = 100
    config.patch_counts = [0]*config.num_centers

    a = ProtoConv2d(2, 16, 3, 1, 1)
    inp = torch.rand(4,2,28,28)
    cent = torch.rand(128, 18)
    a(inp, (cent, temp, config))

