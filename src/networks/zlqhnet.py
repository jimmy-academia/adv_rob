import torch
import torch.nn as nn

from networks.test_time import BasicBlock

class ZLQHNet(nn.Module):
    def __init__(self, args):
        super(ZLQHNet, self).__init__()
        self.args = args
        self.patch_numel = args.channels * args.patch_size * args.patch_size
        self.grid_size = args.image_size // args.patch_size

        self.zero_predictor = nn.Sequential(*self._make_layers(1, 2, 1))
        self.linear_predictor = nn.Sequential(*self._make_layers(3, 4, 1))
        # self.quadratic_predictor = nn.Sequential(*self._make_layers(3, 8, 2))
        self.high_predictor = nn.Sequential(*self._make_layers(args.vocab_size, 8, 3))
        self.embedding = nn.Embedding(args.vocab_size, self.patch_numel)  # T, 12
        self.softmax = nn.Softmax(-1)

        ## move patch from stack back to image
        self.inv_patcher = nn.ConvTranspose2d(self.patch_numel, args.channels, args.patch_size, args.patch_size)
        self._initialize_weights()
        for param in self.inv_patcher.parameters():
            param.requires_grad = False

        # coefficients
        self.lin_coeff = lambda x: [1 - 2 * (i / (x - 1)) for i in range(x)]
        self.lambda_lin = 0.3 # 0.1
        self.lambda_high = 0.1 # 0.01

    def _make_layers(self, _final_channel, init_channel=16, additional_layers=3, min_output=0):
        total_reduction = self.args.patch_size; reduction_done = 1

        _layers = []
        in_dim = self.args.channels
        out_dim = init_channel

        non_lin = nn.Mish()
        final_non_lin = nn.Sigmoid() if min_output == 0 else nn.Tanh()

        while True:
            s = 2 if reduction_done < total_reduction else 1
            _layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=s, padding=1))
            _layers.append(non_lin)
            in_dim = out_dim
            out_dim = out_dim * 2
            reduction_done *= 2
            if s == 1: 
                for __ in range(additional_layers):
                    _layers.append(BasicBlock(in_dim, in_dim, nn.BatchNorm2d, 1))
                break

        _layers.append(nn.Conv2d(in_dim, _final_channel, kernel_size=1, stride=1))
        _layers.append(final_non_lin)
        return _layers

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
        args = self.args
        z = self.zero_predictor(x)  # bs, 1, 16, 16
        z = z.repeat(1, args.channels, 1, 1)
        z = z.repeat_interleave(args.patch_size, dim=2).repeat_interleave(args.patch_size, dim=3) 

        # self.lin_coeff = lambda x: [1 - 2 * (i / (x - 1)) for i in range(x)]
        l = self.linear_predictor(x) # bs, 3, 16, 16
        a = torch.stack(list(map(lambda f: l[:, 0]*f, self.lin_coeff(args.patch_size))))
        a = a.permute(1,0,2,3).permute(0,2,1,3).mT.flatten(-2).repeat_interleave(args.patch_size, dim=1)
        a = a.unsqueeze(1).repeat(1, args.channels,1,1)

        b = torch.stack(list(map(lambda f: l[:, 1]*f, self.lin_coeff(args.patch_size))))
        b = b.permute(1,0,2,3).permute(0,3,2,1).flatten(-2).repeat_interleave(args.patch_size, dim=1).mT
        b = b.unsqueeze(1).repeat(1, args.channels,1,1)

        if args.channels != 1:
            c = torch.stack(list(map(lambda f: l[:, 2]*f, self.lin_coeff(args.channels)))).permute(1,0,2,3)
            c = c.repeat_interleave(args.patch_size, dim=2).repeat_interleave(args.patch_size, dim=3)
            l = (a+b+c)/3
        else:
            # l[:, 2] is not used
            l = (a+b)/2
        l = l / l.max() if l.max() > 1 else l 

        # [(a+b+c, -a+b+c), (a+b+0, -a+b+0), (a+b-c, -a+b-c)]
        # [(a-b+c, -a-b+c), (a-b+0, -a-b+0), (a-b-c, -a-b-c)]

        # q = self.quadratic_predictor(x)
        # q = q.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)

        x = self.high_predictor(x)     # bs, T, 16, 16
        x = x.permute(0, 2,3,1)     # bs, 16, 16, T
        x = x.view(x.size(0), -1, x.size(-1))   # bs, 256, T
        x = self.softmax(x)
        x = torch.matmul(x, self.embedding.weight)  # bs, 256, 12
        x = self.inverse(x)         # bs, 3, 32, 32

        return z + self.lambda_lin*l + self.lambda_high*x
    
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

'''
l = torch.rand(2, 3, 2, 2)

torch.stack(list(map(lambda f: l[:, 2]*f, lin_coeff(3)))).permute(1,0,2,3).repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)

torch.stack(list(map(lambda f: l[:, 0]*f, lin_coeff(2)))).permute(1,0,2,3).permute(0,2,1,3).mT.flatten(-2).repeat_interleave(2, dim=1).unsqueeze(1).repeat(1,3,1,1)

torch.stack(list(map(lambda f: l[:, 1]*f, lin_coeff(2)))).permute(1,0,2,3).permute(0,3,2,1).flatten(-2).repeat_interleave(2, dim=1).mT.unsqueeze(1).repeat(1,3,1,1)


torch.stack(list(map(lambda f: l[:, 0]*f, lin_coeff(2)))).permute(3,1,2,0)
.permute(1,0,2,3).permute(0,3,2,1).flatten(-2).repeat_interleave(2, dim=1).mT

'''