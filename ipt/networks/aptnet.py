import torch
import torch.nn as nn
from utils import check

class APTNet(nn.Module):
    def __init__(self, args, _layers=[16, 64]):
        # in_size, out_size, hidden_layers
        # in_size: args.channels
        # out_size: args.vocab_size
        super(APTNet, self).__init__()
        self.args = args
        self.patch_numel = args.channels * args.patch_size * args.patch_size
        self.num_patch_width = args.image_size // args.patch_size

        self.conv_layers = []
        in_size = args.channels
        for out_size in _layers:
            s = 2 if len(self.conv_layers) == 0 else 1
            self.conv_layers.append(nn.Conv2d(in_size, out_size, kernel_size=3, stride=s, padding=1))
            self.conv_layers.append(nn.ReLU())
            in_size = out_size
        self.conv_layers.append(nn.Conv2d(in_size, args.vocab_size, kernel_size=1, stride=1))
        self.conv_layers = nn.Sequential(*self.conv_layers)

        self.embedding = nn.Embedding(args.vocab_size, self.patch_numel)
        self.softmax = nn.Softmax(-1)
        self.relu = nn.ReLU()

        self.inv_patcher = nn.ConvTranspose2d(self.patch_numel, args.channels, args.patch_size, args.patch_size)
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
        # self.patcher.weight.data = identity_filter
        # self.patcher.bias.data.zero_()

        self.inv_patcher.weight.data = identity_filter
        self.inv_patcher.bias.data.zero_()
    
    def inverse(self, x):
        assert len(x.shape) in [2, 3], 'Input shape should be either [batch_size, num_patches, patch_numel] or [num_patches, patch_numel]'
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        batch_size = x.size(0)
        x = x.contiguous().view(batch_size, self.num_patch_width, self.num_patch_width, self.patch_numel)  
        x = x.permute(0, 3, 1, 2)
        x = self.inv_patcher(x)
        return x

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.permute(0, 2,3,1)
        x = x.view(x.size(0), -1, x.size(-1))
        x = torch.matmul(x, self.embedding.weight)
        x = self.inverse(x)
        return x
    
