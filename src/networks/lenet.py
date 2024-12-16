import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, args):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(args.channels, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Calculate flattened size after conv layers
        fc1_input_dim = self._compute_fc1_input_dim(args.image_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(fc1_input_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def _compute_fc1_input_dim(self, image_size):
        # Simulate passing a dummy input through the conv layers
        dummy_input = torch.zeros(1, 3, image_size, image_size)
        x = F.max_pool2d(F.relu(self.conv1(dummy_input)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
