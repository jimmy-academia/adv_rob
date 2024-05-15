import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import check

class Tokenizer(nn.Module):
    def __init__(self, in_size, out_size, num_hidden_layer=0):
        super().__init__()
        hidden_dim = [8, 32, 128]
        # hidden_dim = [8, 16, 32, 64, 128]
        _layers = []
        for i in range(num_hidden_layer):
            _layers.append(nn.Linear(in_size, hidden_dim[i]))
            _layers.append(nn.ReLU())
            in_size = hidden_dim[i]
        _layers.append(nn.Linear(in_size, out_size))

        self.main_module = nn.Sequential(*_layers)

    def forward(self, x):
        return self.main_module(x)


class TokClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = Tokenizer(args.patch_size**2, args.vocabulary_size, args.num_hidden_layer)
        self.embedding = nn.Embedding(args.vocabulary_size, args.embed_size)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)       
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) 
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)      
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2)
        self.fc = nn.Linear(64, args.num_classes)
        self.split_patch = lambda x: x.unfold(2, args.patch_size, args.patch_size).unfold(3, args.patch_size, args.patch_size).contiguous().view(x.size(0), -1, args.patch_size*args.patch_size*x.size(1))

    def conv_output(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2], self.args.patch_size, self.args.patch_size)
        x = x.permute(0, 1, 3, 2, 4)
        x = x.contiguous().view(x.shape[0], 1, self.args.image_size, self.args.image_size)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)       
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

    def forward(self, x):
        x = self.embedding(x)
        x = self.conv_output(x)
        return x

    def inference_image(self, x):
        x = self.split_patch(x)
        token_probability = self.tokenizer(x)
        x = torch.matmul(token_probability, self.embedding.weight)
        w = self.args.image_size // self.args.patch_size
        x = x.view(x.shape[0], w, w, x.shape[-1])
        x = self.conv_output(x)
        return x
        