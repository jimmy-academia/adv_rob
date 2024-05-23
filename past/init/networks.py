import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import check

class Tokenizer(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.main_module = nn.Sequential(
            nn.Linear(in_size, 8),
            nn.ReLU(),
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, out_size),
        )
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x):
        return self.linear(x)
        # return self.main_module(x)
    
# a model that splits images into patches, tokenize them, and then read from an embedding table
# class ImageEmbedder(nn.Module):
#     def __init__(self, tokenizer, patch_size=2, T=1024, embed_size=128):
#         super().__init__()
#         self.patch_size = patch_size
#         self.T = T
#         self.embed_size = embed_size
#         self.tokenizer = tokenizer
#         self.embedding = nn.Embedding(T, embed_size)
    
#     def forward(self, x):
#         h, w = x.shape[1:]
#         patches = []
#         for y in range(0, h, self.patch_size):
#             for x in range(0, w, self.patch_size):
#                 window = x[:, y:y+self.patch_size, x:x+self.patch_size].flatten()
#                 patches.append(window)
#         patches = torch.stack(patches).to(0)
#         embedded = self.embedding(patches)
#         return embedded

class TokClassifier(nn.Module):
    def __init__(self, T, embed_size=8, num_classes=10, patch_size=2):
        super().__init__()
        
        self.tokenizer = Tokenizer(patch_size**2, T)
        self.embedding = nn.Embedding(T, embed_size)
        self.conv1 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)  # Output: (batch_size, 16, 14, 14)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)       # Output: (batch_size, 16, 7, 7)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # Output: (batch_size, 32, 7, 7)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.split_patch = lambda x: x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).contiguous().view(x.size(0), -1, patch_size*patch_size*x.size(1))

        # self.splite_patch = lambda x: x.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size).contiguous().view(-1, patch_size*patch_size*x.size(0))

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 3, 1, 2)  # Change data shape from (batch_size, 14, 14, 8) to (batch_size, 8, 14, 14)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def inference_image(self, x):
        x = self.split_patch(x)
        token_probability = self.tokenizer(x)
        x = torch.matmul(token_probability, self.embedding.weight)
        x = x.view(x.shape[0], 14, 14, x.shape[-1])
        x = x.permute(0, 3, 1, 2)
        ## rest is same as forward
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        