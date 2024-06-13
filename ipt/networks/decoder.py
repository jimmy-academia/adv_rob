import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        hidden_dim = args.config['train']['decoder_channels']
        self.conv1 = nn.Conv2d(args.channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, args.channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x