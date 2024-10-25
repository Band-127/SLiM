import torch
from torch import nn
from torch.nn import functional as F
from .repvgg import (
    create_RepVGG_A1,
    create_RepVGG_A1_modified,
)  # latter same as Efficient LoFTR

backbone_file = "maff/backbone/repvgg/pretrained_ckpt/RepVGG-A1-train.pth"


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


class RepVGG_Feature_Extractor(nn.Module):
    def __init__(self, config):
        super(RepVGG_Feature_Extractor, self).__init__()

        backbone = create_RepVGG_A1(False, False)

        if "pretrained" in config["BACKBONE_TYPE"]:
            checkpoint = torch.load(backbone_file)
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            ckpt = {
                k.replace("module.", ""): v for k, v in checkpoint.items()
            }  # strip the names
            backbone.load_state_dict(ckpt)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = (
            backbone.stage0,
            backbone.stage1,
            backbone.stage2,
            backbone.stage3,
            backbone.stage4,
        )

        # Add a 1-channel grayscale to 3-channel grayscale conv
        self.conv1to3 = nn.Conv2d(1, 3, kernel_size=1, stride=1, bias=False)
        with torch.no_grad():
            self.conv1to3.weight.fill_(1.0)

    def switch_to_deploy(self):
        module = self.layer0
        if hasattr(module, "switch_to_deploy"):
            module.switch_to_deploy()
        for modules in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for module in modules:
                if hasattr(module, "switch_to_deploy"):
                    module.switch_to_deploy()

    def forward(self, x):
        features = []
        x = self.conv1to3(x)

        out = self.layer0(x)
        features.append(out)

        for module in self.layer1:
            out = module(out)
        features.append(out)

        for module in self.layer2:
            out = module(out)
        features.append(out)

        for module in self.layer3:
            out = module(out)
        features.append(out)

        for module in self.layer4:
            out = module(out)
        features.append(out)

        return features  # [1/2, 1/4, 1/8, 1/16, 1/32]


class RepVGG_Feature_Extractor_with_FPN(nn.Module):
    def __init__(self, config):
        super(RepVGG_Feature_Extractor_with_FPN, self).__init__()

        backbone = create_RepVGG_A1(False, False)

        if "pretrained" in config["BACKBONE_TYPE"]:
            checkpoint = torch.load(backbone_file)
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            ckpt = {
                k.replace("module.", ""): v for k, v in checkpoint.items()
            }  # strip the names
            backbone.load_state_dict(ckpt)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = (
            backbone.stage0,
            backbone.stage1,
            backbone.stage2,
            backbone.stage3,
            backbone.stage4,
        )

        # Add a 1-channel grayscale to 3-channel grayscale conv
        self.conv1to3 = nn.Conv2d(1, 3, kernel_size=1, stride=1, bias=False)
        with torch.no_grad():
            self.conv1to3.weight.fill_(1.0)

        self.dims = (64, 64, 128, 256, 1280)
        # FPN
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

    def switch_to_deploy(self):
        module = self.layer0
        if hasattr(module, "switch_to_deploy"):
            module.switch_to_deploy()
        for modules in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for module in modules:
                if hasattr(module, "switch_to_deploy"):
                    module.switch_to_deploy()

    def forward(self, x):
        features = []
        x = self.conv1to3(x)

        out = self.layer0(x)
        features.append(out)

        for module in self.layer1:
            out = module(out)
        features.append(out)

        for module in self.layer2:
            out = module(out)
        features.append(out)

        for module in self.layer3:
            out = module(out)
        features.append(out)

        for module in self.layer4:
            out = module(out)
        features.append(out)

        # FPN
        fpn_features = [
            self.fpn_layers[-1](features[-1])
        ]  # smallest scale feature no need to fuse with other scales
        for i in range(len(features) - 2, -1, -1):
            higher_feature = F.interpolate(
                fpn_features[0], scale_factor=2, mode="bilinear", align_corners=True
            )
            current_feature = self.fpn_layers[i](features[i])
            fused_feature = self.fpn_fuse[i - 1](higher_feature + current_feature)
            fpn_features.insert(0, fused_feature)

        return fpn_features  # [1/2, 1/4, 1/8, 1/16, 1/32]


class RepVGG_Feature_Extractor_cropped(nn.Module):
    def __init__(self, config):
        super(RepVGG_Feature_Extractor_cropped, self).__init__()

        backbone = create_RepVGG_A1_modified(False, False)  # four stage

        self.layer0, self.layer1, self.layer2, self.layer3 = (
            backbone.stage0,
            backbone.stage1,
            backbone.stage2,
            backbone.stage3,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def switch_to_deploy(self):
        module = self.layer0
        if hasattr(module, "switch_to_deploy"):
            module.switch_to_deploy()
        for modules in [self.layer1, self.layer2, self.layer3]:
            for module in modules:
                if hasattr(module, "switch_to_deploy"):
                    module.switch_to_deploy()

    def forward(self, x):
        features = []

        out = self.layer0(x)

        for module in self.layer1:
            out = module(out)
        features.append(out)

        for module in self.layer2:
            out = module(out)
        features.append(out)

        for module in self.layer3:
            out = module(out)
        features.append(out)

        return features  # [1/2, 1/4, 1/8]


class RepVGG_Feature_Extractor_pretrained_cropped(nn.Module):
    def __init__(self, config):
        super(RepVGG_Feature_Extractor_pretrained_cropped, self).__init__()

        backbone = create_RepVGG_A1(False, False)

        if "pretrained" in config["BACKBONE_TYPE"]:
            checkpoint = torch.load(backbone_file)
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            ckpt = {
                k.replace("module.", ""): v for k, v in checkpoint.items()
            }  # strip the names
            backbone.load_state_dict(ckpt)

        self.layer0, self.layer1, self.layer2 = (
            backbone.stage0,
            backbone.stage1,
            backbone.stage2,
        )
        # Add a 1-channel grayscale to 3-channel grayscale conv
        self.conv1to3 = nn.Conv2d(1, 3, kernel_size=1, stride=1, bias=False)
        with torch.no_grad():
            self.conv1to3.weight.fill_(1.0)

    def switch_to_deploy(self):
        module = self.layer0
        if hasattr(module, "switch_to_deploy"):
            module.switch_to_deploy()
        for modules in [self.layer1, self.layer2]:
            for module in modules:
                if hasattr(module, "switch_to_deploy"):
                    module.switch_to_deploy()

    def forward(self, x):
        features = []
        x = self.conv1to3(x)

        out = self.layer0(x)
        features.append(out)

        for module in self.layer1:
            out = module(out)
        features.append(out)

        for module in self.layer2:
            out = module(out)
        features.append(out)

        return features  # [1/2, 1/4, 1/8]
