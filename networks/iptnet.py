import torch
import torch.nn as nn

class APTNet(nn.Module):
    '''
    APTNet outputs (image_size // patch_size)^2 x vocab_size
    by processing the full image as a whole. 
    Alternative (IPTNet) is to process each patch individually.
    '''
    def __init__(self, args, base_filter_channels=16):
        # in_dim, out_dim, hidden_layers
        # first in_dim: args.channels
        # last out_dim: args.vocab_size
        super(APTNet, self).__init__()
        self.args = args
        self.patch_numel = args.channels * args.patch_size * args.patch_size
        self.num_patch_width = args.image_size // args.patch_size

        ## Automatically determine the strides based on patch_size

        total_reduction = args.patch_size; reduction_done = 1

        _layers = []
        in_dim = args.channels
        out_dim = base_filter_channels

        while True:
            s = 2 if reduction_done < total_reduction else 1
            _layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=s, padding=1))
            _layers.append(nn.ReLU())
            in_dim = out_dim
            out_dim = out_dim * 2
            reduction_done *= 2
            if s == 1: # one more with stride = 1; can increase in future
                break

        _layers.append(nn.Conv2d(in_dim, args.vocab_size, kernel_size=1, stride=1))
        self.conv_layers = nn.Sequential(*_layers)

        ## todo: add more conv2d with stride=1
        # bs, 3, image_size, image_size => bs, T, image_size//patch_size, image_size//patch_size

        self.embedding = nn.Embedding(args.vocab_size, self.patch_numel)  # T, 12
        self.softmax = nn.Softmax(-1)
        self.relu = nn.ReLU()

        ## move patch from stack back to image
        self.inv_patcher = nn.ConvTranspose2d(self.patch_numel, args.channels, args.patch_size, args.patch_size)
        self._initialize_weights()
        for param in self.inv_patcher.parameters():
            param.requires_grad = False

    def _initialize_weights(self):
        # Set the weights for an identity operation

        k, c, w = self.patch_numel, self.args.channels, self.args.patch_size
        identity_filter = torch.zeros((k,c,w,w))
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
        x = self.conv_layers(x)     # bs, T, 16, 16
        x = x.permute(0, 2,3,1)     # bs, 16, 16, T
        x = x.view(x.size(0), -1, x.size(-1))   # bs, 256, T
        x = torch.matmul(x, self.embedding.weight)  # bs, 256, 12
        x = self.inverse(x)         # bs, 3, 32, 32
        return x
    
