from .resnet import ResNet18_2_4_8, ResNet18_2_4_8_16
from .resnet_modified import ResNet18_2_4_8_modified, ResNet18_2_4_8_16_modified
from .resnet_pretrained import ResNet18Pretrained, ResNet18Pretrained_with_FPN
from .vssm import (
    VMamba_Feature_Extractor,
    VMamba_Feature_Extractor_with_FPN,
    VMamba_Feature_Extractor_cropped,
    VMamba_Feature_Extractor_modified,
    VMamba_Feature_Extractor_cropped_FPN
)
from .repvgg import (
    RepVGG_Feature_Extractor,
    RepVGG_Feature_Extractor_with_FPN,
    RepVGG_Feature_Extractor_cropped,
    RepVGG_Feature_Extractor_pretrained_cropped,
)


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
    elif config["BACKBONE_TYPE"] == "ResNet18_pretrained_FPN":
        return ResNet18Pretrained_with_FPN(config=config)
    # VMamba
    elif config["BACKBONE_TYPE"] == "VMamba_T":
        return VMamba_Feature_Extractor(config=config)
    elif config["BACKBONE_TYPE"] == "VMamba_S":
        return VMamba_Feature_Extractor(config=config)
    elif config["BACKBONE_TYPE"] == "VMamba_B":
        return VMamba_Feature_Extractor(config=config)
    elif config["BACKBONE_TYPE"] == "VMamba_T_modified":
        return VMamba_Feature_Extractor_modified(config=config)
    elif config["BACKBONE_TYPE"] == "VMamba_S_modified":
        return VMamba_Feature_Extractor_modified(config=config)
    elif config["BACKBONE_TYPE"] == "VMamba_B_modified":
        return VMamba_Feature_Extractor_modified(config=config)
    elif config["BACKBONE_TYPE"] == "VMamba_T_FPN":
        return VMamba_Feature_Extractor_with_FPN(config=config)
    elif config["BACKBONE_TYPE"] == "VMamba_S_FPN":
        return VMamba_Feature_Extractor_with_FPN(config=config)
    elif config["BACKBONE_TYPE"] == "VMamba_B_FPN":
        return VMamba_Feature_Extractor_with_FPN(config=config)
    elif config["BACKBONE_TYPE"] == "VMamba_T_cropped":
        return VMamba_Feature_Extractor_cropped(config=config)
    elif config["BACKBONE_TYPE"] == "VMamba_S_cropped":
        return VMamba_Feature_Extractor_cropped(config=config)
    elif config["BACKBONE_TYPE"] == "VMamba_B_cropped":
        return VMamba_Feature_Extractor_cropped(config=config)
    elif config["BACKBONE_TYPE"] == "VMamba_T_cropped_FPN":
        return VMamba_Feature_Extractor_cropped_FPN(config=config)
    # RepVGG
    elif (
        config["BACKBONE_TYPE"] == "RepVGG"
        or config["BACKBONE_TYPE"] == "RepVGG_pretrained"
    ):
        return RepVGG_Feature_Extractor(config=config)
    elif (
        config["BACKBONE_TYPE"] == "RepVGG_FPN"
        or config["BACKBONE_TYPE"] == "RepVGG_pretrained_FPN"
    ):
        return RepVGG_Feature_Extractor_with_FPN(config=config)
    elif config["BACKBONE_TYPE"] == "RepVGG_cropped":
        return RepVGG_Feature_Extractor_cropped(config=config)
    elif config["BACKBONE_TYPE"] == "RepVGG_pretrained_cropped":
        return RepVGG_Feature_Extractor_pretrained_cropped(config=config)
    else:
        raise ValueError(
            f"MODEL.BACKBONE.BACKBONE_TYPE {config['BACKBONE_TYPE']} not supported."
        )
