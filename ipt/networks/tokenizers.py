import torch.nn as nn

class MLPTokenizer(nn.Module):
    def __init__(self, in_size, out_size, hidden_layers):
        super().__init__()
        _layers = []
        for _size in hidden_layers:
            _layers.append(nn.Linear(in_size, _size))
            _layers.append(nn.ReLU())
            in_size = _size
        _layers.append(nn.Linear(in_size, out_size))
        self.main_module = nn.Sequential(*_layers)

    def forward(self, x):
        return self.main_module(x)
