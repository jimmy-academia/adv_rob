import torch.nn as nn
import torch.nn.functional as F

class Dummy(nn.Module):
    def __init__(self, iptnet, classifier):
        super(Dummy, self).__init__()
        self.iptnet = iptnet
        self.classifier = classifier
    
    def forward(self, x):
        x = self.iptnet(x)
        x = self.classifier(x)
        return x

class SmallClassifier(nn.Module):
    def __init__(self, args):
        # assume input 32x32
        super(SmallClassifier, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(args.channels, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64, args.num_classes) # 64 = 16 * 2 * 2
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
    
class AptSizeClassifier(nn.Module):
    def __init__(self, args):
        # assume input 32x32
        super(AptSizeClassifier, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(args.channels, 8, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1) 
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) 
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 2 * 2, 128)  # Assumes input image size is 32x32
        self.fc2 = nn.Linear(128, args.num_classes)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.conv1(x)) # 64x8x8
        x = self.pool(F.relu(self.conv2(x))) # 64x8x8
        x = self.pool(F.relu(self.conv3(x))) # 128x2x2

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F


class smallVAENet(nn.Module):
    def __init__(self, args):
        super(smallVAENet, self).__init__()
        self.args = args
        self.encoder = nn.Sequential(
            nn.Conv2d(args.channels, args.vocab_size, kernel_size=3, stride=2, padding=1),  # [B, 16, H/2, W/2]
            nn.ReLU(),
            # nn.Conv2d(16, args.vocab_size, kernel_size=1, stride=1),  # [B, vocab_size, H/2, W/2]
            # nn.ReLU()
        )
        latent_dim = args.latent_dim if hasattr(args, 'latent_dim') else 4
        
        # Assuming args.vocab_size is the latent dimension
        self.fc_mu = nn.Linear(args.vocab_size * (args.image_size // 2) * (args.image_size // 2), latent_dim)
        self.fc_logvar = nn.Linear(args.vocab_size * (args.image_size // 2) * (args.image_size // 2), latent_dim)
        
        self.fc_decode = nn.Linear(latent_dim, args.vocab_size * (args.image_size // 2) * (args.image_size // 2))
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(args.vocab_size, args.channels, kernel_size=3, stride=2, padding=1,  output_padding=1),  # [B, 64, H/2, W/2]
            # nn.ReLU(),
            # nn.ConvTranspose2d(64, 16, kernel_size=3, stride=1, padding=1),  # [B, 16, H/2, W/2]
            # nn.ReLU(),
            # nn.ConvTranspose2d(64, args.channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, channels, H, W]
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), self.args.vocab_size, self.args.image_size // 2, self.args.image_size // 2)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)





class VAENet(nn.Module):
    def __init__(self, args):
        super(VAENet, self).__init__()
        self.args = args
        self.encoder = nn.Sequential(
            nn.Conv2d(args.channels, 16, kernel_size=3, stride=2, padding=1),  # [B, 16, H/2, W/2]
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, H/2, W/2]
            nn.ReLU(),
            nn.Conv2d(64, args.vocab_size, kernel_size=1, stride=1),  # [B, vocab_size, H/2, W/2]
            nn.ReLU()
        )
        latent_dim = args.latent_dim if hasattr(args, 'latent_dim') else 128
        
        # Assuming args.vocab_size is the latent dimension
        self.fc_mu = nn.Linear(args.vocab_size * (args.image_size // 2) * (args.image_size // 2), latent_dim)
        self.fc_logvar = nn.Linear(args.vocab_size * (args.image_size // 2) * (args.image_size // 2), latent_dim)
        
        self.fc_decode = nn.Linear(latent_dim, args.vocab_size * (args.image_size // 2) * (args.image_size // 2))
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(args.vocab_size, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, H/2, W/2]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 16, kernel_size=3, stride=1, padding=1),  # [B, 16, H/2, W/2]
            nn.ReLU(),
            nn.ConvTranspose2d(16, args.channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, channels, H, W]
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), self.args.vocab_size, self.args.image_size // 2, self.args.image_size // 2)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)


