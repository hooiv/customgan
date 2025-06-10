import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.vgg_layers = nn.Sequential(*list(vgg)[:16]).eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        self.resize = resize
    def forward(self, x, y):
        if self.resize:
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            y = nn.functional.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)
        return nn.functional.l1_loss(self.vgg_layers(x), self.vgg_layers(y))
