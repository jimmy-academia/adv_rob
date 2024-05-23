import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from utils import check


class IPTResnet(nn.Module):
    def __init__(self, args):
        super(IPTResnet, self).__init__()
        self.args = args
        self.resnet = self.get_resnet_model(num_classes=args.vocabulary_size, channels=args.input_channels)
        self.tokenembedder = CompactTokenEmbedder(args.image_size, args.patch_size, args.vocabulary_size)
        self.tokenembedder.set_stage('full')

    def forward(self, x):
        x = self.tokenembedder(x)
        x = self.resnet(x)
        return x
    
    @staticmethod
    def get_resnet_model(num_classes=10, channels=3):
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

# class Splitter(nn.Module):
#     def __init__(self, patch_size, image_size):
#         super(Splitter, self).__init__()
#         self.patch_size = patch_size
#         self.image_size = image_size

#     def forward(self, x):
#         return x.view(x.size(0), -1, x.size(1)*self.patch_size)

#     def reverse(self, x):
#         return x.view(x.size(0), -1, self.image_size, self.image_size)
        

## compact design: input -> token -> embedding (same shape as input)


class CompactTokenEmbedder(nn.Module):
    def __init__(self, image_size, patch_size, vocab_size, hidden_dim=128):
        super().__init__()
        self.tokenizer = nn.Sequential(
            nn.Linear(patch_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        self.embedding = nn.Embedding(vocab_size, patch_size)
        self.stage = 'full'
        self.vocab_size = vocab_size
        self.patch_size = patch_size
        self.image_size = image_size

    def set_stage(self, stage):
        options = ['token', 'embedding', 'full']
        if stage not in options:
            raise ValueError(f'Invalid stage, choose from {options}')
        self.stage = stage

    def forward(self, x):
        '''
        stage: token, embedding, full
            |- token: input image output token
            |- embedding: input token output embedding
            |- full: input image output embedding
        '''
        batch_size = x.size(0)
        if self.stage == 'embedding':
            # input tokens
            # x.shape = batch, tok_num
            x = self.embedding(x)
        else:
            # x.shape = batch, C, input_width, input_width
            # x = self.splitter(x)
            x = x.view(x.size(0), -1, x.size(1)*self.patch_size)
            token_probability = self.tokenizer(x)
            if self.stage == 'token':
                # only output token
                if self.attackable:
                    return token_probability.view(-1, self.vocab_size)
                else:
                    return torch.argmax(token_probability, dim=2).view(-1)
            elif self.stage == 'full':
                # output embedding, to be returned to original shape
                if self.attackable:
                    x = torch.matmul(token_probability, self.embedding.weight)
                else:
                    tok_images = torch.argmax(token_probability, dim=2)
                    tok_images = tok_images.view(batch_size, -1)
                    x = self.embedding(tok_images)

        # return to original shape
        # x = self.splitter.reverse(x)
        x = x.view(x.size(0), -1, self.image_size, self.image_size)
        return x

class BPDRTokenEmbedder(CompactTokenEmbedder):
    def __init__(self, image_size, patch_size, vocab_size, hidden_dim=128):
        super().__init__(image_size, patch_size, vocab_size, hidden_dim)
    
    def forward()