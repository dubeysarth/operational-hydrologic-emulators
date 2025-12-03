import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, dim_in, dim_out, layer_sizes, dropout):
        super(Embedding, self).__init__()
        layers = []
        dim_prev = dim_in
        for dim_h in layer_sizes:
            layers.append(nn.Linear(dim_prev, dim_h))
            layers.append(nn.Tanh())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim_prev = dim_h
        layers.append(nn.Linear(dim_prev, dim_out))
        layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)