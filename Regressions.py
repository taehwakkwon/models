import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        
        super(LinearRegression,self).__init__()
        
        self.linear = nn.Linear(in_features,out_features)

    def forward(self,x):
        return self.linear(x)


class Logistic(nn.Module):
    def __init__(self,in_features,out_features):
        # super function. It inherits from nn.Module and we can access everythink in nn.Module
        super(LogisticRegression,self).__init__()
        # Linear function.
        self.linear = nn.Linear(in_features,out_features)
        self.relu = nn.ReLu()

    def forward(self,x):
        x = self.linear(x)
        x = relu(x)
        return x