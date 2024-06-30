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
