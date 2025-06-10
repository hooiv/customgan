import torch
import torch.nn as nn

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss()
    def forward(self, x, y):
        return self.loss(x, y)
