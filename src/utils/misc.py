import torch
from torch import nn
from torch.nn import functional as F


class Unsqueeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(self.dim)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )
        x = x.permute(0, 3, 1, 2)
        return x


class Alpha(nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha], dtype=torch.float32))

    def forward(self, x):
        return self.alpha * x


class Upsample(nn.Module):
    def __init__(self, scale: float = 2.0) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x0, x1):
        B = x0.size(0)
        x = torch.cat([x0, x1], dim=0)
        x = F.interpolate(
            x, scale_factor=self.scale, mode="bilinear", align_corners=False
        )
        x0 = x[:B]
        x1 = x[B:]
        return x0, x1


def create_grid(row_indices: torch.Tensor, col_indices: torch.Tensor) -> torch.Tensor:
    """
    Create a grid of shape (M, W, W, 2).

    Args:
        row_indices (torch.Tensor): A tensor of shape (M, W) representing row indices.
        col_indices (torch.Tensor): A tensor of shape (M, W) representing column indices.

    Returns:
        torch.Tensor: A grid of shape (M, W, W, 2). 2: (x, y)
    """
    M, W = row_indices.shape  # Get the number of samples and the window size
    x = col_indices.view(M, W, 1).repeat(1, 1, W)  # (M, W, W)
    y = row_indices.view(M, 1, W).repeat(1, W, 1)  # (M, W, W)
    return torch.stack((x, y), dim=-1)  # (M, W, W, 2)


class CudaTimer:
    """
    Usage:
        with CudaTimer("process_name") as timer:
            # run your code here
        dt = timer.elapsed_time  # s
    """

    def __init__(self, process_name="", show=False):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.process_name = process_name
        self.show = show

    def __enter__(self):
        self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_event.record()
        torch.cuda.synchronize()
        self.elapsed_time = (
            self.start_event.elapsed_time(self.end_event) / 1000
        )  # ms -> s
        if self.show:
            self._print_time()

    def _print_time(self):
        process_str = f"[{self.process_name}] " if self.process_name else ""
        if self.elapsed_time < 1:
            print(f"{process_str}Elapsed time: {self.elapsed_time * 1000:.2f} ms")
        elif self.elapsed_time < 60:
            print(f"{process_str}Elapsed time: {self.elapsed_time:.2f} s")
        else:
            minutes, seconds = divmod(self.elapsed_time, 60)
            print(f"{process_str}Elapsed time: {int(minutes)} min {seconds:.2f} s")
