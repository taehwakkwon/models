import torch
import torch.nn as nn

cfg = {
    'VGG11' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'VGG13' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'VGG16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'VGG19' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):
    def __init__(self, in_features, out_features, architectures, act, use_bn):
        super(VGG, self).__init__()
        
        if act == 'ReLU':
            self.act = nn.ReLU()
        elif act == 'Sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'Tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError('no valid activation function selected!')
        
        self.in_features = in_features[0]
        self.use_bn = use_bn
        
        
        self.out_features = out_features
        self.architectures = cfg[architectures]
        
        self.layers = []
        
        for arch in architectures:
            if arch == 'M':
                self.layers.append(nn.MaxPool2d(2, 2))                
            else:
                self.layers.append(nn.Conv2d(self.in_features, arch, 3, 1, 1))
                if self.use_bn:
                    self.layers.append(nn.BatchNorm2d(arch))
                    
                self.layers.append(self.act)
                self.in_features = arch
                
        
        self.layers = nn.Sequential(*self.layers)
        
        self.fts = self.get_fts(in_features)
        
        self.classifier = [nn.Linear(self.fts, 4096),
                           self.act,
                           nn.Linear(4096, 4096), 
                           self.act,
                           nn.Linear(4096, 10)]

        self.classifier = nn.Sequential(*self.classifier)
        
    def get_fts(self, in_features):
        fts = self.layers(torch.ones(1,*in_features))
        return int(np.prod(fts.size()[1:]))
        
    def forward(self, x):
        x = self.layers(x)
        
        return self.classifier(x.view(-1, self.fts))