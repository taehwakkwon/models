import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features, out_features, hid_dims, act):
        super(MultiLayerPerceptron, self).__init__()
        
        if act == 'ReLU':
            self.act = nn.ReLU()
        elif act == 'Sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'Tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError('no valid activation function selected!')
        
        
        self.in_features = in_features
        self.out_features = out_features
        self.hid_dims = hid_dims
        
        self.layers = []
        
        for hid_dim in hid_dims:
            self.layers.append(nn.Linear(in_features, hid_dim))
            self.layers.append(self.act)
            in_features = hid_dim
        else:
            self.layers.append(nn.Linear(in_features, out_features))
        
        self.layers = nn.Sequential(*self.layers)
        
        
    def forward(self, x):
        return self.layers(x)