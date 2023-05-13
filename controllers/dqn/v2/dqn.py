import torch
import torch.nn as nn

# Define a Deep Q-Network (DQN) class, which inherits from PyTorch's nn.Module class.
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(input_dim, 64)  # input_layer
        self.fc2 = nn.Linear(64, 64)  # hidden_layer
        self.fc3 = nn.Linear(64, output_dim)  # output_layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x
