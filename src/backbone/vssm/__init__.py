import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .vmamba import VSSM
from src.utils.any_input_identity import AnyInputIdentity

VMAMBA_T_PATH = "src/backbone/vssm/pretrained_ckpt/vssm1_tiny_0230s_ckpt_epoch_264.pth"
VMAMBA_S_PATH = "src/backbone/vssm/pretrained_ckpt/vssm1_small_0229s_ckpt_epoch_240.pth"
VMAMBA_B_PATH = "src/backbone/vssm/pretrained_ckpt/vssm1_base_0229s_ckpt_epoch_225.pth"


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


class VMamba_Feature_Extractor(nn.Module):
    def __init__(self, config):
        super(VMamba_Feature_Extractor, self).__init__()
        if "VMamba_T" in config["BACKBONE_TYPE"]:
            self.backbone = (
                build_pretrained_VMamba_T(config)
                if config["VMAMBA_PRETRAINED"]
                else build_VMamba_T(config)
            )
        elif "VMamba_S" in config["BACKBONE_TYPE"]:
            self.backbone = (
                build_pretrained_VMamba_S(config)
                if config["VMAMBA_PRETRAINED"]
                else build_VMamba_S(config)
            )
        elif "VMamba_B" in config["BACKBONE_TYPE"]:
            self.backbone = (
                build_pretrained_VMamba_B(config)
                if config["VMAMBA_PRETRAINED"]
                else build_VMamba_B(config)
            )
        else:  # Default to Tiny
            self.backbone = (
                build_pretrained_VMamba_T(config)
                if config["VMAMBA_PRETRAINED"]
                else build_VMamba_T(config)
            )

        # remove classifier
        del self.backbone.classifier

        # Add a 1-channel grayscale to 3-channel grayscale conv
        self.conv1to3 = conv1x1(1, 3)
        with torch.no_grad():
            self.conv1to3.weight.fill_(1.0)

        # Add a pixel shuffle layer to extract 1/2 scale features
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x0, x1):
        # Concatenate x0 and x1 along the batch dimension
        x = torch.cat([x0, x1], dim=0)
        features = []
        # patch embed
        x = self.conv1to3(x)
        x = self.backbone.patch_embed(x)
        features.append(self.pixel_shuffle(x))
        # pos embed
        if self.backbone.pos_embed is not None:
            pos_embed = (
                self.backbone.pos_embed.permute(0, 2, 3, 1)
                if not self.backbone.channel_first
                else self.backbone.pos_embed
            )
            x = x + pos_embed
        # forward
        for layer in self.backbone.layers:
            x = layer.blocks(x)
            features.append(x)
            x = layer.downsample(x)
        # Split the features back into two separate batches
        features_0 = [f[: x0.size(0)] for f in features]
        features_1 = [f[x0.size(0) :] for f in features]

        return features_0, features_1

    def flops(self, shape=(3, 224, 224), verbose=True):
        return self.backbone.flops(shape, verbose)


class VMamba_Feature_Extractor_modified(nn.Module):
    def __init__(self, config):
        super(VMamba_Feature_Extractor_modified, self).__init__()
        if "VMamba_T" in config["BACKBONE_TYPE"]:
            self.backbone = build_modified_VMamba_T(config)
        elif "VMamba_S" in config["BACKBONE_TYPE"]:
            self.backbone = build_modified_VMamba_S(config)
        elif "VMamba_B" in config["BACKBONE_TYPE"]:
            self.backbone = build_modified_VMamba_B(config)
        else:  # Default to Tiny
            self.backbone = build_modified_VMamba_T(config)

        # remove classifier and last layer
        self.backbone.layers.pop(-1)
        del self.backbone.classifier

    def forward(self, x0, x1):
        # Concatenate x0 and x1 along the batch dimension
        x = torch.cat([x0, x1], dim=0)
        features = []
        # patch embed
        x = self.backbone.patch_embed(x)
        # pos embed
        if self.backbone.pos_embed is not None:
            pos_embed = (
                self.backbone.pos_embed.permute(0, 2, 3, 1)
                if not self.backbone.channel_first
                else self.backbone.pos_embed
            )
            x = x + pos_embed
        # forward
        for layer in self.backbone.layers:
            x = layer(x)
            features.append(x)
        # Split the features back into two separate batches
        features_0 = [f[: x0.size(0)] for f in features]
        features_1 = [f[x0.size(0) :] for f in features]

        return features_0, features_1

    def flops(self, shape=(3, 224, 224), verbose=True):
        return self.backbone.flops(shape, verbose)


class VMamba_Feature_Extractor_with_FPN(nn.Module):
    def __init__(self, config):
        super(VMamba_Feature_Extractor_with_FPN, self).__init__()
        if "VMamba_T" in config["BACKBONE_TYPE"]:
            self.backbone = (
                build_pretrained_VMamba_T(config)
                if config["VMAMBA_PRETRAINED"]
                else build_VMamba_T(config)
            )
        elif "VMamba_S" in config["BACKBONE_TYPE"]:
            self.backbone = (
                build_pretrained_VMamba_S(config)
                if config["VMAMBA_PRETRAINED"]
                else build_VMamba_S(config)
            )
        elif "VMamba_B" in config["BACKBONE_TYPE"]:
            self.backbone = (
                build_pretrained_VMamba_B(config)
                if config["VMAMBA_PRETRAINED"]
                else build_VMamba_B(config)
            )
        else:  # Default to Tiny
            self.backbone = (
                build_pretrained_VMamba_T(config)
                if config["VMAMBA_PRETRAINED"]
                else build_VMamba_T(config)
            )
        # remove classifier
        del self.backbone.classifier

        # Add a 1-channel grayscale to 3-channel grayscale conv
        self.conv1to3 = conv1x1(1, 3)
        with torch.no_grad():
            self.conv1to3.weight.fill_(1.0)

        # Add a pixel shuffle layer to extract 1/2 scale features
        self.pixel_shuffle = nn.PixelShuffle(2)

        # FPN
        self.fpn_in_channels = [int(self.backbone.dims[0] // 4), *self.backbone.dims]
        self.final_outconv = conv1x1(self.fpn_in_channels[-1], self.fpn_in_channels[-1])
        self.lateral_convs = nn.ModuleList(
            [
                conv1x1(in_channels, self.fpn_in_channels[i + 1])
                for i, in_channels in enumerate(self.fpn_in_channels[:-1])
            ]
        )
        self.output_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        self.fpn_in_channels[i + 1],
                        self.fpn_in_channels[i + 1],
                        kernel_size=3,
                        padding=1,
                        groups=self.fpn_in_channels[i + 1],
                    ),
                    LayerNorm2d(self.fpn_in_channels[i + 1]),
                    nn.GELU(),
                    nn.Conv2d(
                        self.fpn_in_channels[i + 1],
                        self.fpn_in_channels[i],
                        kernel_size=1,
                        padding=0,
                    ),
                )
                for i in range(len(self.fpn_in_channels) - 1)
            ]
        )

        # Initialization
        with torch.no_grad():
            for m in self.final_outconv.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            for m in self.lateral_convs.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            for m in self.output_convs.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x0, x1):
        # Concatenate x0 and x1 along the batch dimension
        x = torch.cat([x0, x1], dim=0)
        features = []
        # patch embed
        x = self.conv1to3(x)
        x = self.backbone.patch_embed(x)
        features.append(self.pixel_shuffle(x))
        # pos embed
        if self.backbone.pos_embed is not None:
            pos_embed = (
                self.backbone.pos_embed.permute(0, 2, 3, 1)
                if not self.backbone.channel_first
                else self.backbone.pos_embed
            )
            x = x + pos_embed
        # forward
        for layer in self.backbone.layers:
            x = layer.blocks(x)
            features.append(x)
            x = layer.downsample(x)

        # Split the features back into two separate batches
        features_0 = [f[: x0.size(0)] for f in features]
        features_1 = [f[x0.size(0) :] for f in features]

        # FPN 0
        prev_features = self.final_outconv(features_0[-1])
        fpn_features_0 = [prev_features]

        for i in range(len(features_0) - 2, -1, -1):
            higher_feature = F.interpolate(
                fpn_features_0[0], scale_factor=2, mode="bilinear", align_corners=False
            )
            current_feature = self.lateral_convs[i](features_0[i])
            fused_feature = self.output_convs[i](higher_feature + current_feature)
            fpn_features_0.insert(0, fused_feature)
        # FPN 1
        prev_features = self.final_outconv(features_1[-1])
        fpn_features_1 = [prev_features]

        for i in range(len(features_1) - 2, -1, -1):
            higher_feature = F.interpolate(
                fpn_features_1[0], scale_factor=2, mode="bilinear", align_corners=False
            )
            current_feature = self.lateral_convs[i](features_1[i])
            fused_feature = self.output_convs[i](higher_feature + current_feature)
            fpn_features_1.insert(0, fused_feature)

        return fpn_features_0, fpn_features_1

    def flops(self, shape=(3, 224, 224), verbose=True):
        return self.backbone.flops(shape, verbose)


class VMamba_Feature_Extractor_cropped(nn.Module):
    def __init__(self, config):
        super(VMamba_Feature_Extractor_cropped, self).__init__()
        if "VMamba_T" in config["BACKBONE_TYPE"]:
            self.backbone = (
                build_pretrained_VMamba_T(config)
                if config["VMAMBA_PRETRAINED"]
                else build_VMamba_T(config)
            )
        elif "VMamba_S" in config["BACKBONE_TYPE"]:
            self.backbone = (
                build_pretrained_VMamba_S(config)
                if config["VMAMBA_PRETRAINED"]
                else build_VMamba_S(config)
            )
        elif "VMamba_B" in config["BACKBONE_TYPE"]:
            self.backbone = (
                build_pretrained_VMamba_B(config)
                if config["VMAMBA_PRETRAINED"]
                else build_VMamba_B(config)
            )
        else:  # Default to Tiny
            self.backbone = (
                build_pretrained_VMamba_T(config)
                if config["VMAMBA_PRETRAINED"]
                else build_VMamba_T(config)
            )

        # remove classifier and last two layer
        del self.backbone.classifier
        self.backbone.layers.pop(-1)
        self.backbone.layers.pop(-1)

        # Add a 1-channel grayscale to 3-channel grayscale conv
        self.conv1to3 = conv1x1(1, 3)
        with torch.no_grad():
            self.conv1to3.weight.fill_(1.0)

        # Add a pixel shuffle layer to extract 1/2 scale features
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x0, x1):
        # Concatenate x0 and x1 along the batch dimension
        x = torch.cat([x0, x1], dim=0)
        features = []
        # patch embed
        x = self.conv1to3(x)
        x = self.backbone.patch_embed(x)
        features.append(self.pixel_shuffle(x))
        # pos embed
        if self.backbone.pos_embed is not None:
            pos_embed = (
                self.backbone.pos_embed.permute(0, 2, 3, 1)
                if not self.backbone.channel_first
                else self.backbone.pos_embed
            )
            x = x + pos_embed
        # forward
        for layer in self.backbone.layers:
            x = layer.blocks(x)
            features.append(x)
            x = layer.downsample(x)
        # Split the features back into two separate batches
        features_0 = [f[: x0.size(0)] for f in features]
        features_1 = [f[x0.size(0) :] for f in features]

        return features_0, features_1

    def flops(self, shape=(3, 224, 224), verbose=True):
        return self.backbone.flops(shape, verbose)


class VMamba_Feature_Extractor_cropped_concat(nn.Module):
    def __init__(self, config):
        super(VMamba_Feature_Extractor_cropped_concat, self).__init__()
        if "VMamba_T" in config["BACKBONE_TYPE"]:
            self.backbone = (
                build_pretrained_VMamba_T(config)
                if config["VMAMBA_PRETRAINED"]
                else build_VMamba_T(config)
            )
        elif "VMamba_S" in config["BACKBONE_TYPE"]:
            self.backbone = (
                build_pretrained_VMamba_S(config)
                if config["VMAMBA_PRETRAINED"]
                else build_VMamba_S(config)
            )
        elif "VMamba_B" in config["BACKBONE_TYPE"]:
            self.backbone = (
                build_pretrained_VMamba_B(config)
                if config["VMAMBA_PRETRAINED"]
                else build_VMamba_B(config)
            )
        else:  # Default to Tiny
            self.backbone = (
                build_pretrained_VMamba_T(config)
                if config["VMAMBA_PRETRAINED"]
                else build_VMamba_T(config)
            )

        # remove classifier and last two layer
        del self.backbone.classifier
        self.backbone.layers.pop(-1)
        self.backbone.layers.pop(-1)

        # Add a 1-channel grayscale to 3-channel grayscale conv
        self.conv1to3 = conv1x1(1, 3)
        with torch.no_grad():
            self.conv1to3.weight.fill_(1.0)

        # Add a pixel shuffle layer to extract 1/2 scale features
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x0, x1):
        features = []
        features_0 = []
        features_1 = []
        # patch embed
        x0, x1 = self.conv1to3(x0), self.conv1to3(x1)
        x0, x1 = self.backbone.patch_embed(x0), self.backbone.patch_embed(x1)
        features_0.append(self.pixel_shuffle(x0))
        features_1.append(self.pixel_shuffle(x1))
        # pos embed
        if self.backbone.pos_embed is not None:
            pos_embed = (
                self.backbone.pos_embed.permute(0, 2, 3, 1)
                if not self.backbone.channel_first
                else self.backbone.pos_embed
            )
            x0 = x0 + pos_embed
            x1 = x1 + pos_embed

        x = torch.concat([x0, x1], dim=-1)  # Concat in W channel
        # forward
        for layer in self.backbone.layers:
            x = layer.blocks(x)
            features.append(x)
            x = layer.downsample(x)
        # Split into two features
        for feature in features:
            # Assuming the feature maps are concatenated along the width dimension
            feature_0, feature_1 = torch.chunk(feature, 2, dim=-1)
            features_0.append(feature_0)
            features_1.append(feature_1)

        return features_0, features_1

    def flops(self, shape=(3, 224, 224), verbose=True):
        return self.backbone.flops(shape, verbose)


class VMamba_Feature_Extractor_cropped_FPN(nn.Module):
    def __init__(self, config):
        super(VMamba_Feature_Extractor_cropped_FPN, self).__init__()
        if "VMamba_T" in config["BACKBONE_TYPE"]:
            self.backbone = (
                build_pretrained_VMamba_T(config)
                if config["VMAMBA_PRETRAINED"]
                else build_VMamba_T(config)
            )
        elif "VMamba_S" in config["BACKBONE_TYPE"]:
            self.backbone = (
                build_pretrained_VMamba_S(config)
                if config["VMAMBA_PRETRAINED"]
                else build_VMamba_S(config)
            )
        elif "VMamba_B" in config["BACKBONE_TYPE"]:
            self.backbone = (
                build_pretrained_VMamba_B(config)
                if config["VMAMBA_PRETRAINED"]
                else build_VMamba_B(config)
            )
        else:  # Default to Tiny
            self.backbone = (
                build_pretrained_VMamba_T(config)
                if config["VMAMBA_PRETRAINED"]
                else build_VMamba_T(config)
            )

        # remove classifier and last two layer
        del self.backbone.classifier
        self.backbone.layers.pop(-1)
        self.backbone.layers.pop(-1)
        self.backbone.dims.pop(-1)
        self.backbone.dims.pop(-1)

        # Add a 1-channel grayscale to 3-channel grayscale conv
        self.conv1to3 = conv1x1(1, 3)
        with torch.no_grad():
            nn.init.constant_(self.conv1to3.weight, 1.0)

        # Add a pixel shuffle layer to extract 1/2 scale features
        self.pixel_shuffle = nn.PixelShuffle(2)

        # FPN
        self.fpn_in_channels = [int(self.backbone.dims[0] // 4), *self.backbone.dims]
        self.final_outconv = conv1x1(self.fpn_in_channels[-1], self.fpn_in_channels[-1])
        self.lateral_convs = nn.ModuleList(
            [
                conv1x1(in_channels, self.fpn_in_channels[i + 1])
                for i, in_channels in enumerate(self.fpn_in_channels[:-1])
            ]
        )
        self.output_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        self.fpn_in_channels[i + 1],
                        self.fpn_in_channels[i + 1],
                        kernel_size=3,
                        padding=1,
                        groups=self.fpn_in_channels[i + 1],
                    ),
                    LayerNorm2d(self.fpn_in_channels[i + 1]),
                    nn.GELU(),
                    nn.Conv2d(
                        self.fpn_in_channels[i + 1],
                        self.fpn_in_channels[i],
                        kernel_size=1,
                        padding=0,
                    ),
                )
                for i in range(len(self.fpn_in_channels) - 1)
            ]
        )

        # Initialization
        with torch.no_grad():
            for m in self.final_outconv.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            for m in self.lateral_convs.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            for m in self.output_convs.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x0, x1):
        # Concatenate x0 and x1 along the batch dimension
        x = torch.cat([x0, x1], dim=0)
        features = []
        # patch embed
        x = self.conv1to3(x)
        x = self.backbone.patch_embed(x)
        features.append(self.pixel_shuffle(x))
        # pos embed
        if self.backbone.pos_embed is not None:
            pos_embed = (
                self.backbone.pos_embed.permute(0, 2, 3, 1)
                if not self.backbone.channel_first
                else self.backbone.pos_embed
            )
            x = x + pos_embed
        # forward
        for layer in self.backbone.layers:
            x = layer.blocks(x)
            features.append(x)
            x = layer.downsample(x)

        # Split the features back into two separate batches
        features_0 = [f[: x0.size(0)] for f in features]
        features_1 = [f[x0.size(0) :] for f in features]

        # FPN 0
        prev_features = self.final_outconv(features_0[-1])
        fpn_features_0 = [prev_features]

        for i in range(len(features_0) - 2, -1, -1):
            higher_feature = F.interpolate(
                fpn_features_0[0], scale_factor=2, mode="bilinear", align_corners=False
            )
            current_feature = self.lateral_convs[i](features_0[i])
            fused_feature = self.output_convs[i](higher_feature + current_feature)
            fpn_features_0.insert(0, fused_feature)
        # FPN 1
        prev_features = self.final_outconv(features_1[-1])
        fpn_features_1 = [prev_features]

        for i in range(len(features_1) - 2, -1, -1):
            higher_feature = F.interpolate(
                fpn_features_1[0], scale_factor=2, mode="bilinear", align_corners=False
            )
            current_feature = self.lateral_convs[i](features_1[i])
            fused_feature = self.output_convs[i](higher_feature + current_feature)
            fpn_features_1.insert(0, fused_feature)

        return fpn_features_0, fpn_features_1

    def flops(self, shape=(3, 224, 224), verbose=True):
        return self.backbone.flops(shape, verbose)


class VMamba_Feature_Extractor_cropped_concat_FPN(nn.Module):
    def __init__(self, config):
        super(VMamba_Feature_Extractor_cropped_concat_FPN, self).__init__()
        if "VMamba_T" in config["BACKBONE_TYPE"]:
            self.backbone = (
                build_pretrained_VMamba_T(config)
                if config["VMAMBA_PRETRAINED"]
                else build_VMamba_T(config)
            )
        elif "VMamba_S" in config["BACKBONE_TYPE"]:
            self.backbone = (
                build_pretrained_VMamba_S(config)
                if config["VMAMBA_PRETRAINED"]
                else build_VMamba_S(config)
            )
        elif "VMamba_B" in config["BACKBONE_TYPE"]:
            self.backbone = (
                build_pretrained_VMamba_B(config)
                if config["VMAMBA_PRETRAINED"]
                else build_VMamba_B(config)
            )
        else:  # Default to Tiny
            self.backbone = (
                build_pretrained_VMamba_T(config)
                if config["VMAMBA_PRETRAINED"]
                else build_VMamba_T(config)
            )

        # remove classifier and last two layer
        del self.backbone.classifier
        self.backbone.layers.pop(-1)
        self.backbone.layers.pop(-1)
        self.backbone.dims.pop(-1)
        self.backbone.dims.pop(-1)

        # Add a 1-channel grayscale to 3-channel grayscale conv
        self.conv1to3 = conv1x1(1, 3)
        with torch.no_grad():
            nn.init.constant_(self.conv1to3.weight, 1.0)

        # Add a pixel shuffle layer to extract 1/2 scale features
        self.pixel_shuffle = nn.PixelShuffle(2)

        # FPN
        self.fpn_in_channels = [int(self.backbone.dims[0] // 4), *self.backbone.dims]
        self.final_outconv = conv1x1(self.fpn_in_channels[-1], self.fpn_in_channels[-1])
        self.lateral_convs = nn.ModuleList(
            [
                conv1x1(in_channels, self.fpn_in_channels[i + 1])
                for i, in_channels in enumerate(self.fpn_in_channels[:-1])
            ]
        )
        self.output_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        self.fpn_in_channels[i + 1],
                        self.fpn_in_channels[i + 1],
                        kernel_size=3,
                        padding=1,
                        groups=self.fpn_in_channels[i + 1],
                    ),
                    LayerNorm2d(self.fpn_in_channels[i + 1]),
                    nn.GELU(),
                    nn.Conv2d(
                        self.fpn_in_channels[i + 1],
                        self.fpn_in_channels[i],
                        kernel_size=1,
                        padding=0,
                    ),
                )
                for i in range(len(self.fpn_in_channels) - 1)
            ]
        )

        # Initialization
        with torch.no_grad():
            for m in self.final_outconv.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            for m in self.lateral_convs.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            for m in self.output_convs.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x0, x1):
        # Concatenate x0 and x1 along the batch dimension
        features = []
        features_0 = []
        features_1 = []
        # patch embed
        x0, x1 = self.conv1to3(x0), self.conv1to3(x1)
        x0, x1 = self.backbone.patch_embed(x0), self.backbone.patch_embed(x1)
        features_0.append(self.pixel_shuffle(x0))
        features_1.append(self.pixel_shuffle(x1))
        # pos embed
        if self.backbone.pos_embed is not None:
            pos_embed = (
                self.backbone.pos_embed.permute(0, 2, 3, 1)
                if not self.backbone.channel_first
                else self.backbone.pos_embed
            )
            x0 = x0 + pos_embed
            x1 = x1 + pos_embed

        x = torch.concat([x0, x1], dim=-1)  # Concat in W channel
        # forward
        for layer in self.backbone.layers:
            x = layer.blocks(x)
            features.append(x)
            x = layer.downsample(x)

        # Split into two features
        for feature in features:
            # Assuming the feature maps are concatenated along the width dimension
            feature_0, feature_1 = torch.chunk(feature, 2, dim=-1)
            features_0.append(feature_0)
            features_1.append(feature_1)

        # FPN 0
        prev_features = self.final_outconv(features_0[-1])
        fpn_features_0 = [prev_features]

        for i in range(len(features_0) - 2, -1, -1):
            higher_feature = F.interpolate(
                fpn_features_0[0], scale_factor=2, mode="bilinear", align_corners=False
            )
            current_feature = self.lateral_convs[i](features_0[i])
            fused_feature = self.output_convs[i](higher_feature + current_feature)
            fpn_features_0.insert(0, fused_feature)
        # FPN 1
        prev_features = self.final_outconv(features_1[-1])
        fpn_features_1 = [prev_features]

        for i in range(len(features_1) - 2, -1, -1):
            higher_feature = F.interpolate(
                fpn_features_1[0], scale_factor=2, mode="bilinear", align_corners=False
            )
            current_feature = self.lateral_convs[i](features_1[i])
            fused_feature = self.output_convs[i](higher_feature + current_feature)
            fpn_features_1.insert(0, fused_feature)

        return fpn_features_0, fpn_features_1

    def flops(self, shape=(3, 224, 224), verbose=True):
        return self.backbone.flops(shape, verbose)


def build_VMamba_T(config):
    model = VSSM(
        depths=[2, 2, 8, 2],
        dims=96,
        drop_path_rate=0.2,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        ssm_d_state=1,
        ssm_ratio=1.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=False,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v05_noz",
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        patch_norm=True,
        norm_layer="ln2d",
        downsample_version="v3",
        patchembed_version="v2",
        use_checkpoint=False,
        posembed=False,
        imgsize=config["INPUT_SIZE"],
    )
    return model


def build_pretrained_VMamba_T(config):
    model = build_VMamba_T(config)
    state_dict = torch.load(VMAMBA_T_PATH)["model"]
    model.load_state_dict(state_dict)
    return model


def build_modified_VMamba_T(config):
    model = VSSM(
        depths=[1, 2, 2, 1],
        dims=24,
        drop_path_rate=0.2,
        patch_size=2,
        in_chans=1,
        num_classes=1000,
        ssm_d_state=1,
        ssm_ratio=1.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=False,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v05_noz",
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        patch_norm=True,
        norm_layer="ln2d",
        downsample_version="v3",
        patchembed_version="v2",
        use_checkpoint=False,
        posembed=False,
        imgsize=config["INPUT_SIZE"],
    )
    return model


def build_VMamba_S(config):
    model = VSSM(
        depths=[2, 2, 20, 2],
        dims=96,
        drop_path_rate=0.3,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        ssm_d_state=1,
        ssm_ratio=1.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=False,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v05_noz",
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        patch_norm=True,
        norm_layer="ln2d",
        downsample_version="v3",
        patchembed_version="v2",
        use_checkpoint=False,
        posembed=False,
        imgsize=config["INPUT_SIZE"],
    )
    return model


def build_pretrained_VMamba_S(config):
    model = build_VMamba_S(config)
    state_dict = torch.load(VMAMBA_S_PATH)["model"]
    model.load_state_dict(state_dict)
    return model


def build_modified_VMamba_S(config):
    model = VSSM(
        depths=[2, 2, 2, 1],
        dims=24,
        drop_path_rate=0.3,
        patch_size=2,
        in_chans=1,
        num_classes=1000,
        ssm_d_state=1,
        ssm_ratio=1.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=False,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v05_noz",
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        patch_norm=True,
        norm_layer="ln2d",
        downsample_version="v3",
        patchembed_version="v2",
        use_checkpoint=False,
        posembed=False,
        imgsize=config["INPUT_SIZE"],
    )
    return model


def build_VMamba_B(config):
    model = VSSM(
        depths=[2, 2, 20, 2],
        dims=128,
        drop_path_rate=0.5,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        ssm_d_state=1,
        ssm_ratio=1.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=False,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v05_noz",
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        patch_norm=True,
        norm_layer="ln2d",
        downsample_version="v3",
        patchembed_version="v2",
        use_checkpoint=False,
        posembed=False,
        imgsize=config["INPUT_SIZE"],
    )
    return model


def build_pretrained_VMamba_B(config):
    model = build_VMamba_B(config)
    state_dict = torch.load(VMAMBA_B_PATH)["model"]
    model.load_state_dict(state_dict)
    return model


def build_modified_VMamba_B(config):
    model = VSSM(
        depths=[2, 2, 2, 1],
        dims=32,
        drop_path_rate=0.5,
        patch_size=2,
        in_chans=1,
        num_classes=1000,
        ssm_d_state=1,
        ssm_ratio=1.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=False,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v05_noz",
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        patch_norm=True,
        norm_layer="ln2d",
        downsample_version="v3",
        patchembed_version="v2",
        use_checkpoint=False,
        posembed=False,
        imgsize=config["INPUT_SIZE"],
    )
    return model
