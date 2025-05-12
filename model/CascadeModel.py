import torch
import torch.nn as nn

class CascadeModel(nn.Module):
    def __init__(self, modelA, modelB):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB

    def forward(self, x):
        x = self.modelA(x)
        x = self.modelB(x)
        return x

    def denoise(self, x):
        x = self.modelA(x)
        return x

    def classifier(self, x):
        x = self.modelB(x)
        return x

class CascadeModelWithBathNorm(nn.Module):
    def __init__(self, modelA, modelB):
        super().__init__()
        self.modelA = modelA
        self.bn = nn.BatchNorm2d(1)
        self.modelB = modelB

    def forward(self, x):
        x = self.modelA(x)
        x = self.bn(x)
        x = self.modelB(x)
        return x

    def denoise(self, x):
        x = self.modelA(x)
        x = self.bn(x)
        return x

    def classifier(self, x):
        x = self.modelB(x)
        return x