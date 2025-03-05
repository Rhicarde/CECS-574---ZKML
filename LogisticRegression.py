import torch
import torch.nn as nn


class LogisticRegressionTorch(nn.Module):
    def __init__(self, weights, bias):
        super(LogisticRegressionTorch, self).__init__()
        self.linear = nn.Linear(len(weights), 1)
        self.linear.weight = nn.Parameter(torch.tensor([weights], dtype=torch.float32))
        self.linear.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    def forward(self, x):
        return torch.sigmoid(self.linear(x))