from .resnet import ResNet18_2_4_8, ResNet18_2_4_8_16


def build_backbone(config):
    if config["BACKBONE_TYPE"] == "ResNet18":
        if config["RESOLUTION"] == (2, 4, 8):
            return ResNet18_2_4_8(config=config)
        elif config["RESOLUTION"] == (2, 4, 8, 16):
            return ResNet18_2_4_8_16(config=config)
        else:
            raise ValueError(f"MODEL.BACKBONE.RESOLUTION {config['RESOLUTION']} not supported.")
    else:
        raise ValueError(f"MODEL.BACKBONE.BACKBONE_TYPE {config['BACKBONE_TYPE']} not supported.")
