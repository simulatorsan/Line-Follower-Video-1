import torch
import torch.nn as nn

class HiddenLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
    def forward(self, x):
        return self.activation(self.linear(x))

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32, hidden_layers=1):
        super(DQN, self).__init__()
        assert hidden_layers >= 0, "hidden_layers must be non-negative"
        if hidden_layers == 0:
            self.net = nn.Linear(state_dim, action_dim)
            return
        
        
        layers = [
            HiddenLayer(state_dim, hidden_dim)
        ]
        for _ in range(hidden_layers-1):
            layers.append(HiddenLayer(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
