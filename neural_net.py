import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, i, h1, h2, o):
        super(NeuralNetwork, self).__init__()
        # Initial weights are randomly selected
        # Inputs to hidden layer linear transformation
        self.layer1 = nn.Linear(i, h1)
        # h1 to h2 linear transformation
        self.layer2 = nn.Linear(h1, h2)
        # h2 to output linear transformation
        self.layer3 = nn.Linear(h2, o)

        # Define relu activation and LogSoftmax output
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.sig(self.layer1(x))
        out = self.sig(self.layer2(out))
        out = self.sig(self.layer3(out))
        return out
