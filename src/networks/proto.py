import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.cluster_centers = torch.rand(args.num_centers, in_channels*kernel_size**2).to(args.device)
        

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
        
            patch_size = self.kernel_size**2 * self.in_channels
            flat_patches = this_patch.transpose(1,2).reshape(-1, patch_size)

            if train:
                # stream update center with this_patch 
                for _ in range(self.args.num_cluster_iterations):
                    # Compute distances from patches to cluster centers
                    distances = torch.cdist(flat_patches, self.cluster_centers)
                    labels = torch.argmin(distances, dim=1)

                # Update cluster centers
                for i in range(self.args.num_centers):
                    cluster_patches = flat_patches[labels == i].detach()
                    if len(cluster_patches) > 0:
                        self.cluster_centers[i] = (self.cluster_centers[i] * self.patch_counts[i] + cluster_patches.mean(dim=0) * len(cluster_patches)) / (len(cluster_patches) + self.patch_counts[i])
                        self.patch_counts[i] += len(cluster_patches)
                
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

