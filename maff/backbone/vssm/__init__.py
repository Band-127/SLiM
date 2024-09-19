import torch
from torch import nn
from .vmamba import VSSM


class VMamba_Feature_Extractor(nn.Module):
    def __init__(self, config):
        super(VMamba_Feature_Extractor, self).__init__()
        if config["BACKBONE_TYPE"] == "VMamba_T":
            self.backbone = build_pretrained_VMamba_T(config)
        elif config["BACKBONE_TYPE"] == "VMamba_S":
            self.backbone = build_pretrained_VMamba_S(config)
        elif config["BACKBONE_TYPE"] == "VMamba_B":
            self.backbone = build_pretrained_VMamba_B(config)
        else:  # Default to Tiny
            self.backbone = build_pretrained_VMamba_T(config)

        # remove classifier
        del self.backbone.classifier

        # Add a 1-channel grayscale to 3-channel grayscale conv
        self.conv1to3 = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        with torch.no_grad():
            self.conv1to3.weight.fill_(1.0)

    def forward(self, x):
        # patch embed
        x = self.conv1to3(x)
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
        features = []
        for layer in self.backbone.layers:
            x = layer.blocks(x)
            features.append(x)
            x = layer.downsample(x)

        return features

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
        "maff/backbone/vssm/pretrained_ckpt/vssm1_tiny_0230s_ckpt_epoch_264.pth"
    )["model"]
    model.load_state_dict(state_dict)
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
        "maff/backbone/vssm/pretrained_ckpt/vssm1_small_0229s_ckpt_epoch_240.pth"
    )["model"]
    model.load_state_dict(state_dict)
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
        "maff/backbone/vssm/pretrained_ckpt/vssm1_base_0229s_ckpt_epoch_225.pth"
    )["model"]
    model.load_state_dict(state_dict)
    return model
