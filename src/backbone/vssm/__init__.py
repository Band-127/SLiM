import torch
from torch import nn
from .vmamba import VSSM
import torch.nn.functional as F

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
            self.backbone = build_pretrained_VMamba_T(config)
        elif "VMamba_S" in config["BACKBONE_TYPE"]:
            self.backbone = build_pretrained_VMamba_S(config)
        elif "VMamba_B" in config["BACKBONE_TYPE"]:
            self.backbone = build_pretrained_VMamba_B(config)
        else:  # Default to Tiny
            self.backbone = build_pretrained_VMamba_T(config)

        # remove classifier
        del self.backbone.classifier

        # Add a 1-channel grayscale to 3-channel grayscale conv
        self.conv1to3 = conv1x1(1, 3)
        with torch.no_grad():
            self.conv1to3.weight.fill_(1.0)

        # Add a pixel shuffle layer to extract 1/2 scale features
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
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

        return features

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
        del self.backbone.classifier
        self.backbone.layers.pop(-1)

    def forward(self, x):
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
            x = layer.blocks(x)
            x = layer.downsample(x)
            features.append(x)

        return features

    def flops(self, shape=(3, 224, 224), verbose=True):
        return self.backbone.flops(shape, verbose)


class VMamba_Feature_Extractor_with_FPN(nn.Module):
    def __init__(self, config):
        super(VMamba_Feature_Extractor_with_FPN, self).__init__()
        if "VMamba_T" in config["BACKBONE_TYPE"]:
            self.backbone = build_pretrained_VMamba_T(config)
        elif "VMamba_S" in config["BACKBONE_TYPE"]:
            self.backbone = build_pretrained_VMamba_S(config)
        elif "VMamba_B" in config["BACKBONE_TYPE"]:
            self.backbone = build_pretrained_VMamba_B(config)
        else:  # Default to Tiny
            self.backbone = build_pretrained_VMamba_T(config)

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

        return fpn_features

    def flops(self, shape=(3, 224, 224), verbose=True):
        return self.backbone.flops(shape, verbose)


class VMamba_Feature_Extractor_cropped(nn.Module):
    def __init__(self, config):
        super(VMamba_Feature_Extractor_cropped, self).__init__()
        if "VMamba_T" in config["BACKBONE_TYPE"]:
            self.backbone = build_pretrained_VMamba_T(config)
        elif "VMamba_S" in config["BACKBONE_TYPE"]:
            self.backbone = build_pretrained_VMamba_S(config)
        elif "VMamba_B" in config["BACKBONE_TYPE"]:
            self.backbone = build_pretrained_VMamba_B(config)
        else:  # Default to Tiny
            self.backbone = build_pretrained_VMamba_T(config)

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

    def forward(self, x):
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

        return features

    def flops(self, shape=(3, 224, 224), verbose=True):
        return self.backbone.flops(shape, verbose)


class VMamba_Feature_Extractor_cropped_FPN(nn.Module):
    def __init__(self, config):
        super(VMamba_Feature_Extractor_cropped_FPN, self).__init__()
        if "VMamba_T" in config["BACKBONE_TYPE"]:
            self.backbone = build_pretrained_VMamba_T(config)
        elif "VMamba_S" in config["BACKBONE_TYPE"]:
            self.backbone = build_pretrained_VMamba_S(config)
        elif "VMamba_B" in config["BACKBONE_TYPE"]:
            self.backbone = build_pretrained_VMamba_B(config)
        else:  # Default to Tiny
            self.backbone = build_pretrained_VMamba_T(config)

        # remove classifier and last two layer
        del self.backbone.classifier
        self.backbone.layers.pop(-1)
        self.backbone.layers.pop(-1)
        self.backbone.dims.pop(-1)
        self.backbone.dims.pop(-1)

        # Add a 1-channel grayscale to 3-channel grayscale conv
        self.conv1to3 = conv1x1(1, 3)
        with torch.no_grad():
            self.conv1to3.weight.fill_(1.0)

        # Add a pixel shuffle layer to extract 1/2 scale features
        self.pixel_shuffle = nn.PixelShuffle(2)

        # FPN
        self.fpn_in_channels = [int(self.backbone.dims[0] // 4), *self.backbone.dims]
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

        # FPN
        fpn_features = [
            self.fpn_layers[-1](features[-1])
        ]  # smallest scale feature no need to fuse with other scales
        for i in range(len(features) - 2, -1, -1):
            higher_feature = F.interpolate(
                fpn_features[0], scale_factor=2, mode="bilinear", align_corners=False
            )
            current_feature = self.fpn_layers[i](features[i])
            fused_feature = self.fpn_fuse[i - 1](higher_feature + current_feature)
            fpn_features.insert(0, fused_feature)

        return fpn_features

    def flops(self, shape=(3, 224, 224), verbose=True):
        return self.backbone.flops(shape, verbose)


def build_pretrained_VMamba_T(config):
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
    state_dict = torch.load(
        VMAMBA_T_PATH
    )["model"]
    model.load_state_dict(state_dict)
    return model


def build_modified_VMamba_T(config):
    model = VSSM(
        depths=[2, 2, 8, 2],
        dims=96,
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


def build_pretrained_VMamba_S(config):
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
    state_dict = torch.load(
        VMAMBA_S_PATH
    )["model"]
    model.load_state_dict(state_dict)
    return model


def build_modified_VMamba_S(config):
    model = VSSM(
        depths=[2, 2, 20, 2],
        dims=96,
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


def build_pretrained_VMamba_B(config):
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
    state_dict = torch.load(
        VMAMBA_B_PATH
    )["model"]
    model.load_state_dict(state_dict)
    return model


def build_modified_VMamba_B(config):
    model = VSSM(
        depths=[2, 2, 20, 2],
        dims=128,
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
