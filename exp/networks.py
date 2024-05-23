import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet_model(num_classes=10, channels=3):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class Tokenizer(nn.Module):
    def __init__(self, in_size, out_size, num_hidden_layer):
        super().__init__()
        hidden_dim = [8, 32, 128]
        _layers = []
        for i in range(num_hidden_layer):
            _layers.append(nn.Linear(in_size, hidden_dim[i]))
            _layers.append(nn.ReLU())
            in_size = hidden_dim[i]
        _layers.append(nn.Linear(in_size, out_size))

        self.main_module = nn.Sequential(*_layers)

    def forward(self, x):
        return self.main_module(x)

class IPTResnet(nn.Module):
    def __init__(self, args):
        super(IPTResnet, self).__init__()
        self.args = args
        self.tokenizer = Tokenizer(args.patch_size*args.channels, args.vocab_size, args.num_hidden_layer)
        self.embedding = nn.Embedding(args.vocab_size, args.patch_size*args.channels)
        self.classifier = get_resnet_model(num_classes=args.vocab_size, channels=args.channels)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(-1, x.size(1)*self.args.patch_size)
        x = self.tokenizer(x)
        x = torch.matmul(x, self.embedding.weight)
        x = x.view(batch_size, -1, self.args.image_size, self.args.image_size)
        x = self.classifier(x)
        return x
        
    def from_tokens(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1, self.args.image_size, self.args.image_size)
        x = self.classifier(x)
        return x

    def inference(self, x):
        x = x.view(x.size(0), -1, x.size(1)*self.args.patch_size)
        x = self.tokenizer(x)
        x = torch.argmax(x, dim=2) 
        x = self.from_tokens(x)
        return x

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--vocab_size', type=int, default=1024)
    parser.add_argument('--num_hidden_layer', type=int, default=2)
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=32)
    args = parser.parse_args()
    
    model = IPTResnet(args)
    model.to(0)
    model.tokenizer.load_state_dict(torch.load('ckpt/tokenizer_mnist_8_1024.pth'))
    x = torch.randn(1024, 1, 32, 32)
    x = x.to(0)
    x.requires_grad = True
    out = model(x)
    print(out)
    loss = out.sum()
    loss.backward()
    print('pass')        
