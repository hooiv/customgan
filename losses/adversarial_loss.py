import torch
import torch.nn as nn

class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
    def forward(self, pred, target):
        return self.loss(pred, target)
