import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet_model(num_classes=10, channels=3):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class IPTResnet(nn.Module):
    def __init__(self, args, PatchMaker, Tokenizer):
        super(IPTResnet, self).__init__()
        self.args = args
        self.patcher = PatchMaker
        self.patch_numel = args.channels * args.patch_size * args.patch_size
        self.tokenizer = Tokenizer
        self.embedding = nn.Embedding(args.vocab_size, self.patch_numel)
        self.classifier = get_resnet_model(num_classes=args.num_classes, channels=args.channels)
        self.softmax = nn.Softmax(-1)
        self.tau = 10

    def tokenize_image(self, x, flat=False):
        x = self.patcher(x, flat)  # (batch_size, num_patches, patch_numel)
        x = self.tokenizer(x) # (batch_size, num_patches, vocab_size)
        return x
    
    def patch_embed(self, p):
        p = self.tokenizer(p)
        p = self.softmax(p * self.tau)
        p = torch.matmul(p, self.embedding.weight)
        return p

    def forward(self, x):
        x = self.tokenize_image(x)
        x = self.softmax(x * self.tau)
        x = torch.matmul(x, self.embedding.weight)
        x = self.patcher.inverse(x)
        x = self.classifier(x)
        return x
    
    def visualize_tok_image(self, img):
        tok_image = self.tokenize_image(img.unsqueeze(0)).argmax(2)
        tok_image = tok_image.view(self.args.num_patches_width, self.args.num_patches_width)
        print(tok_image)

