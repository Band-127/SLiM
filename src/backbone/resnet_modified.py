import torch.nn as nn


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
    )


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.ln1 = nn.LayerNorm(planes)  # LayerNorm replacing BatchNorm
        self.ln2 = nn.LayerNorm(planes)  # LayerNorm replacing BatchNorm
        self.relu = nn.ReLU()

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = conv1x1(in_planes, planes, stride=stride)
            self.ln_downsample = nn.LayerNorm(planes)

    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.ln1(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # LayerNorm
        y = self.relu(y)
        y = self.conv2(y)
        y = self.ln2(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # LayerNorm

        if self.downsample is not None:
            x = self.downsample(x)
            x = self.ln_downsample(x.permute(0, 2, 3, 1)).permute(
                0, 3, 1, 2
            )  # LayerNorm

        return self.relu(x + y)


class ResNet18_2_4_8_modified(nn.Module):
    """
    Modified ResNet18 with Layer Normalization.
    """

    def __init__(self, config):
        super(ResNet18_2_4_8_modified, self).__init__()
        # Config
        block = BasicBlock
        block_dims = config["LAYER_DIMS"]
        initial_dim = block_dims[0]

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(
            1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.ln1 = nn.LayerNorm(initial_dim)  # LayerNorm replacing BatchNorm
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # Modified ResNet Backbone
        x0 = self.conv1(x)
        x0 = self.ln1(x0.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # LayerNorm
        x0 = self.relu(x0)
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8

        return [x1, x2, x3]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ResNet18_2_4_8_16_modified(nn.Module):
    """
    Modified ResNet18 with Layer Normalization, output resolution are 1/16, 1/8, 1/4 and 1/2.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super(ResNet18_2_4_8_16_modified, self).__init__()
        # Config
        block = BasicBlock
        block_dims = config["LAYER_DIMS"]
        initial_dim = block_dims[0]

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(
            1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.ln1 = nn.LayerNorm(initial_dim)  # LayerNorm replacing BatchNorm
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8
        self.layer4 = self._make_layer(block, block_dims[3], stride=2)  # 1/16

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # Modified ResNet Backbone
        x0 = self.conv1(x)
        x0 = self.ln1(x0.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # LayerNorm
        x0 = self.relu(x0)
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8
        x4 = self.layer4(x3)  # 1/16

        return [x1, x2, x3, x4]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
