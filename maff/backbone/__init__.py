from .resnet import ResNet18_2_4_8, ResNet18_2_4_8_16
from .resnet_modified import ResNet18_2_4_8_modified, ResNet18_2_4_8_16_modified
from .resnet_pretrained import ResNet18Pretrained
from .vssm import VMamba_Feature_Extractor


def build_backbone(config):
    # ResNet18
    if config["BACKBONE_TYPE"] == "ResNet18":
        if config["RESOLUTION"] == (2, 4, 8):
            return ResNet18_2_4_8(config=config)
        elif config["RESOLUTION"] == (2, 4, 8, 16):
            return ResNet18_2_4_8_16(config=config)
        else:
            raise ValueError(
                f"MODEL.BACKBONE.RESOLUTION {config['RESOLUTION']} not supported."
            )
    # ResNet18_modified
    elif config["BACKBONE_TYPE"] == "ResNet18_modified":
        if config["RESOLUTION"] == (2, 4, 8):
            return ResNet18_2_4_8_modified(config=config)
        elif config["RESOLUTION"] == (2, 4, 8, 16):
            return ResNet18_2_4_8_16_modified(config=config)
    # ResNet18_pretrained
    elif config["BACKBONE_TYPE"] == "ResNet18_pretrained":
        return ResNet18Pretrained()
    # VMamba
    elif config["BACKBONE_TYPE"] == "VMamba_T":
        return VMamba_Feature_Extractor(config=config)
    elif config["BACKBONE_TYPE"] == "VMamba_S":
        return VMamba_Feature_Extractor(config=config)
    elif config["BACKBONE_TYPE"] == "VMamba_B":
        return VMamba_Feature_Extractor(config=config)
    else:
        raise ValueError(
            f"MODEL.BACKBONE.BACKBONE_TYPE {config['BACKBONE_TYPE']} not supported."
        )
