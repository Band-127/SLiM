from .convmamba import ConvVMamba


def build_backbone(config):
    return ConvVMamba(config=config)
