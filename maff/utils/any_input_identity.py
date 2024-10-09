from torch import nn


class AnyInputIdentity(nn.Module):
    """
    Placeholder for input length > 1
    """

    def __init__(self, *args, **kwargs) -> None:
        super(AnyInputIdentity, self).__init__()

    def forward(self, *args):
        return args
