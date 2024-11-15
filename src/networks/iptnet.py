import torch
import torch.nn as nn

from networks.test_time import BasicBlock

class APTNet(nn.Module):
    '''
    APTNet outputs (image_size // patch_size)^2 x vocab_size
    by processing the full image as a whole. 
    Alternative (IPTNet) is to process each patch individually.
    '''
    def __init__(self, args, base_filter_channels=16, additional_layers=3):
        # in_dim, out_dim, hidden_layers
        # first in_dim: args.channels
        # last out_dim: args.vocab_size
        super(APTNet, self).__init__()
        self.args = args
        self.patch_numel = args.channels * args.patch_size * args.patch_size
        self.grid_size = args.image_size // args.patch_size

        ## Automatically determine the strides based on patch_size

        total_reduction = args.patch_size; reduction_done = 1

        _layers = []
        in_dim = args.channels
        out_dim = base_filter_channels

        # non_lin = nn.ReLU()
        non_lin = nn.Sigmoid()

        while True:
            s = 2 if reduction_done < total_reduction else 1
            _layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=s, padding=1))
            _layers.append(non_lin)
            in_dim = out_dim
            out_dim = out_dim * 2
            reduction_done *= 2
            if s == 1: 
                # no size reduction and skip connection
                for __ in range(additional_layers):
                    _layers.append(BasicBlock(in_dim, in_dim, nn.BatchNorm2d, 1))
                    # _layers.append(nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1))
                    # _layers.append(non_lin)
                break

        _layers.append(nn.Conv2d(in_dim, args.vocab_size, kernel_size=1, stride=1))
        self.predictor = nn.Sequential(*_layers)

        ## todo: add more conv2d with stride=1
        # bs, 3, image_size, image_size => bs, T, image_size//patch_size, image_size//patch_size

        self.embedding = nn.Embedding(args.vocab_size, self.patch_numel)  # T, 12
        self.softmax = nn.Softmax(-1)
        # self.relu = nn.ReLU()

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
        x = x.contiguous().view(batch_size, self.grid_size, self.grid_size, self.patch_numel)  
        x = x.permute(0, 3, 1, 2)
        x = self.inv_patcher(x)
        return x

    def forward(self, x):
        x = self.predictor(x)     # bs, T, 16, 16
        x = x.permute(0, 2,3,1)     # bs, 16, 16, T
        x = x.view(x.size(0), -1, x.size(-1))   # bs, 256, T
        x = self.softmax(x)
        x = torch.matmul(x, self.embedding.weight)  # bs, 256, 12
        x = self.inverse(x)         # bs, 3, 32, 32
        return x
    
    def visualize_embeddings(self):

        num_patch_vis = self.grid_size ** 2
        num_vis = (self.args.vocab_size + num_patch_vis - 1)//num_patch_vis

        all_vis_images = []
        for n in range(num_vis):
            batch_embed = self.embedding.weight[num_patch_vis *n :num_patch_vis * (n+1)]
            if len(batch_embed) < num_patch_vis:
                pad_embed = torch.zeros(num_patch_vis - len(batch_embed), *batch_embed.shape[1:]).to(batch_embed.device)
                batch_embed = torch.cat([batch_embed, pad_embed], dim=0)
            vis_image = self.inverse(batch_embed)
            all_vis_images.append(vis_image)

        return torch.cat(all_vis_images)

    def reorder_embedding(self):
        patch_sums = self.embedding.weight.sum(dim=1)
        __, sorted_indices = torch.sort(patch_sums)        
        self.embedding.weight = torch.nn.Parameter(self.embedding.weight[sorted_indices])

    def calc_reg_loss(self):
        weight = self.embedding.weight
        norm_weight = torch.nn.functional.normalize(weight, p=2, dim=1)
        cosine_similarity = torch.matmul(norm_weight, norm_weight.t())
        identity_mask = torch.eye(cosine_similarity.size(0), device=cosine_similarity.device)
        regloss = (cosine_similarity * (1 - identity_mask)).sum() / (cosine_similarity.size(0) * (cosine_similarity.size(0) - 1))

        return regloss