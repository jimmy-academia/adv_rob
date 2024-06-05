import torch
import torch.nn as nn
import torchvision.models as models

import math

class PatchMaker(nn.Module):
    def __init__(self, in_channels, patch_size):
        super(PatchMaker, self).__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        patch_numel = in_channels*patch_size*patch_size
        self.patcher = nn.Conv2d(in_channels, patch_numel, kernel_size=patch_size, stride=patch_size)
        self.inv_patcher = nn.ConvTranspose2d(patch_numel, in_channels, kernel_size=patch_size, stride=patch_size)
        self._initialize_weights()
        for param in self.patcher.parameters():
            param.requires_grad = False
        for param in self.inv_patcher.parameters():
            param.requires_grad = False

    def _initialize_weights(self):
        # Set the weights for an identity operation
        k, c, h, w = self.patcher.weight.shape  # e.g., (12, 3, 2, 2)
        identity_filter = torch.zeros_like(self.patcher.weight)
        for i in range(k):
            channel = i % c    ## 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
            row = i//c % h     ## 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
            col = (i//(c*h))   ## 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 
            identity_filter[i, channel, row, col] = 1

        # Set the manually calculated weights and zero biases
        self.patcher.weight.data = identity_filter
        self.patcher.bias.data.zero_()

        self.inv_patcher.weight.data = identity_filter
        self.inv_patcher.bias.data.zero_()

    def forward(self, x):
        x = self.patcher(x)  # Outputs: [batch_size, 12, 3, 3]
        batch_size, patch_numel, __, __ = x.shape
        x = x.permute(0, 2, 3, 1)  # Put channel as the last dimension
        x = x.contiguous().view(batch_size, -1, patch_numel)  # Flatten grid to sequence
        return x

    def inverse(self, x):
        batch_size, num_patches, patch_numel = x.shape
        w = int(math.sqrt(num_patches))
        assert w**2 == num_patches
        x = x.contiguous().view(batch_size, w, w, patch_numel)  
        x = x.permute(0, 3, 1, 2)
        x = self.inv_patcher(x)
        return x

def get_resnet_model(num_classes=10, channels=3):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# class Tokenizer(nn.Module):
#     def __init__(self, in_size, out_size, num_hidden_layer):
#         super().__init__()
#         hidden_dim = [8, 32, 128]
#         _layers = []
#         for i in range(num_hidden_layer):
#             _layers.append(nn.Linear(in_size, hidden_dim[i]))
#             _layers.append(nn.ReLU())
#             in_size = hidden_dim[i]
#         _layers.append(nn.Linear(in_size, out_size))

#         self.main_module = nn.Sequential(*_layers)

#     def forward(self, x):
#         return self.main_module(x)



#     def _process_input(self, x: torch.Tensor) -> torch.Tensor:
#         n, c, h, w = x.shape
#         p = self.patch_size
#         n_h = h // p
#         n_w = w // p

#         hidden_dim = channel * patch_size * patch_size
#         nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)
#                 self._initialize_weights()

#     def _initialize_weights(self):
#         # Set the weights for an identity operation
#         c, n, h, w = self.conv.weight.shape  # (12, 3, 2, 2)
#         identity_filter = torch.zeros_like(self.conv.weight)
#         for i in range(c):
#             channel = i // (h * w)
#             row = (i % (h * w)) // w
#             col = (i % (h * w)) % w
#             identity_filter[i, channel, row, col] = 1

#         # Set the manually calculated weights and zero biases
#         self.conv.weight.data = identity_filter
#         self.conv.bias.data.zero_()




#         # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
#         x = self.conv_proj(x)
#         # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
#         x = x.reshape(n, self.hidden_dim, n_h * n_w)

#         # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
#         # The self attention layer expects inputs in the format (N, S, E)
#         # where S is the source sequence length, N is the batch size, E is the
#         # embedding dimension
#         x = x.permute(0, 2, 1)

#         return x


# class IPTResnet(nn.Module):
#     def __init__(self, args):
#         super(IPTResnet, self).__init__()
#         self.args = args
#         self.tokenizer = Tokenizer(args.patch_size, args.vocab_size, args.num_hidden_layer)
#         self.embedding = nn.Embedding(args.vocab_size, args.patch_size)
#         self.classifier = get_resnet_model(num_classes=args.vocab_size, channels=args.channels)
        
#     def forward(self, x):
#         batch_size = x.size(0)
#         x = x.view(-1, self.args.patch_size)
#         x = self.tokenizer(x)
#         x = torch.matmul(x, self.embedding.weight)
#         x = x.view(batch_size, -1, self.args.image_size, self.args.image_size)
#         x = self.classifier(x)
#         return x
        
#     def from_tokens(self, x):
#         x = self.embedding(x)
#         x = x.view(x.size(0), -1, self.args.image_size, self.args.image_size)
#         x = self.classifier(x)
#         return x

#     def inference(self, x):
#         x = x.view(x.size(0), -1, self.args.patch_size)
#         x = self.tokenizer(x)
#         x = torch.argmax(x, dim=2) 
#         x = self.from_tokens(x)
#         return x

if __name__ == '__main__':

    patch_maker = PatchMaker(3, 4)
    a = torch.rand(5, 3, 32, 32)
    a.requires_grad = True
    b = patch_maker(a)
    c = patch_maker.inverse(b)

    print(all((a==c).flatten()))
    c.sum().backward()
    print(all((a.grad == 1).flatten()))
    
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--patch_size', type=int, default=8)
    # parser.add_argument('--vocab_size', type=int, default=1024)
    # parser.add_argument('--num_hidden_layer', type=int, default=2)
    # parser.add_argument('--channels', type=int, default=1)
    # parser.add_argument('--image_size', type=int, default=32)
    # args = parser.parse_args()
    
    # model = IPTResnet(args)
    # model.to(0)
    # model.tokenizer.load_state_dict(torch.load('ckpt/tokenizer_mnist_8_1024.pth'))
    # x = torch.randn(1024, 1, 32, 32)
    # x = x.to(0)
    # x.requires_grad = True
    # out = model(x)
    # print(out)
    # loss = out.sum()
    # loss.backward()
    # print('pass')        
