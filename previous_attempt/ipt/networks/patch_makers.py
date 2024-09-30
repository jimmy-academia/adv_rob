import torch
import torch.nn as nn


class DisjointPatchMaker(nn.Module):
    def __init__(self, args):
        super(DisjointPatchMaker, self).__init__()
        self.args = args
        self.patcher = nn.Conv2d(args.channels, args.patch_numel, args.patch_size, args.patch_size)
        self.inv_patcher = nn.ConvTranspose2d(args.patch_numel, args.channels, args.patch_size, args.patch_size)
        self._initialize_weights()
        for param in self.patcher.parameters():
            param.requires_grad = False
        for param in self.inv_patcher.parameters():
            param.requires_grad = False
    
    def _initialize_weights(self):
        # Set the weights for an identity operation
        k, c, w, __ = self.patcher.weight.shape  # e.g., (12, 3, 2, 2)
        identity_filter = torch.zeros_like(self.patcher.weight)
        for i in range(k):
            channel = i % c    ## 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
            row = i//c % w     ## 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
            col = (i//(c*w))   ## 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 
            identity_filter[i, channel, row, col] = 1

        # Set the manually calculated weights and zero biases
        self.patcher.weight.data = identity_filter
        self.patcher.bias.data.zero_()

        self.inv_patcher.weight.data = identity_filter
        self.inv_patcher.bias.data.zero_()

    def forward(self, x, flat=False):
        assert len(x.shape) == 4, 'Input shape should [batch_size, channels, height, width]'
        x = self.patcher(x) # Outputs: [batch_size, patch_numel, num_patches_width, num_patches_width]
        batch_size = x.size(0)
        x = x.permute(0, 2, 3, 1)  # Put patch_numel to the last dimension
        if flat:
            return x.contiguous().view(-1, self.args.patch_numel)  # Outputs: [num_patches, patch_numel]
        else:
            return x.contiguous().view(batch_size, -1, self.args.patch_numel)  # Outputs: [batch_size, num_patches, patch_numel]

    def inverse(self, x):
        assert len(x.shape) in [2, 3], 'Input shape should be either [batch_size, num_patches, patch_numel] or [num_patches, patch_numel]'
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        batch_size = x.size(0)
        x = x.contiguous().view(batch_size, self.args.num_patches_width, self.args.num_patches_width, self.args.patch_numel)  
        x = x.permute(0, 3, 1, 2)
        x = self.inv_patcher(x)
        return x