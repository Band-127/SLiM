import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Pretrained(nn.Module):
    def __init__(self):
        super(ResNet18Pretrained, self).__init__()
        # Load pretrained ResNet18 model
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Add a 1-channel grayscale to 3-channel grayscale conv
        conv1to3 = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        with torch.no_grad():
            conv1to3.weight.fill_(1.0)

        # Remove classification layer
        self.resnet = nn.Sequential(conv1to3, *list(self.resnet.children())[:-2])

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.resnet):
            x = layer(x)
            if i in [5, 6, 7, 8]:  # [1/4, 1/8, 1/16, 1/32]
                features.append(x)

        return features
