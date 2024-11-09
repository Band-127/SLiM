import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels):
        super(LayerNorm2d, self).__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (B, C, H, W)

        Returns:
            torch.Tensor: (B, C, H, W)
        """
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


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
            if i in [3, 5, 6, 7, 8]:  # [1/2, 1/4, 1/8, 1/16, 1/32]
                features.append(x)

        return features


class ResNet18Pretrained_with_FPN(nn.Module):
    def __init__(self, config):
        super(ResNet18Pretrained_with_FPN, self).__init__()
        # Load pretrained ResNet18 model
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Add a 1-channel grayscale to 3-channel grayscale conv
        conv1to3 = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        with torch.no_grad():
            conv1to3.weight.fill_(1.0)

        # Remove classification layer
        self.resnet = nn.Sequential(conv1to3, *list(self.resnet.children())[:-2])

        # FPN
        self.dims = [64, 64, 128, 256, 512]
        self.fpn_in_channels = [int(self.dims[0] // 4), *self.dims]
        self.fpn_out_channels = config["FPN_OUT_CHANNELS"]
        self.fpn_layers = nn.ModuleList(
            [
                conv1x1(in_channels, self.fpn_out_channels)
                for in_channels in self.fpn_in_channels
            ]
        )
        self.fpn_fuse = nn.ModuleList(
            [
                nn.Sequential(
                    conv3x3(self.fpn_out_channels, self.fpn_out_channels),
                    LayerNorm2d(self.fpn_out_channels),
                    nn.LeakyReLU(),
                    conv3x3(self.fpn_out_channels, self.fpn_out_channels),
                )
                for _ in range(len(self.fpn_in_channels) - 1)
            ]
        )

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.resnet):
            x = layer(x)
            if i in [3, 5, 6, 7, 8]:  # [1/2, 1/4, 1/8, 1/16, 1/32]
                features.append(x)

        return features
