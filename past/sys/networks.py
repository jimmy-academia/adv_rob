import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

def get_resnet_model(num_classes=10, pretrained=False):
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

## compact design: input -> token -> embedding (same shape as input)
class CompactTokenEmbedder(nn.Module):
    def __init__(self, input_shape, patch_size, vocab_size, hidden_dim=128):
        super().__init__()
        # input_shape: (C, H, W)
        self.input_channels = input_shape[0]
        self.input_width = input_shape[1]
        self.tok_width = self.input_width // patch_size
        self.patch_size = patch_size
        assert input_shape[1] == input_shape[2], 'Only square images are supported'

        input_size = patch_size**2*self.input_channels
        self.tokenizer = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        self.embedding = nn.Linear(vocab_size, input_size, bias=False)
        self.split_patch = lambda x: x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).contiguous().view(x.size(0), -1, input_size)

        self.stage = 'inference'

    def set_stage(self, stage):
        options = ['train_embedding', 'train_token', 'attack', 'inference']
        if stage not in options:
            raise ValueError(f'Invalid stage, choose from {options}')
        self.stage = stage

    def forward(self, x):
        batch_size = x.size(0)
        if self.stage == 'train_embedding':
            # x.shape = batch, tok_width, tok_width
            x = self.embedding(x)
        else:
            # x.shape = batch, C, input_width, input_width
            x = self.split_patch(x)
            token_probability = self.tokenizer(x)
            if self.stage == 'train_token':
                # return token_prediction
                return torch.argmax(token_probability, dim=2)
            elif self.stage == 'attack':
                x = torch.matmul(token_probability, self.embedding.weight)
            elif self.stage == 'inference':
                tok_images = torch.argmax(token_probability, dim=2)  # Assign the largest element in tok_image to tok_image
                tok_images = tok_images.view(batch_size, self.tok_width, self.tok_width)
                x = self.embedding(tok_images)

        # return to original shape
        x = x.view(batch_size, self.tok_width, self.tok_width, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 3, 2, 4)
        x = x.contiguous().view(batch_size, self.input_channels, self.input_width, self.input_width)
        return x

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
        