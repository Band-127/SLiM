import torch
from torch import nn


class Unsqueeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(self.dim)


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

    # # Initialize an empty grid
    # grid = torch.zeros((M, W, W, 2), device=row_indices.device)

    # for i in range(M):
    #     # Use meshgrid to create the grid for the current sample
    #     col_grid, row_grid = torch.meshgrid(col_indices[i], row_indices[i], indexing='ij')

    #     # Fill the grid with the current sample's grid coordinates
    #     grid[i, :, :, 0] = col_grid
    #     grid[i, :, :, 1] = row_grid

    # return grid
    x = col_indices.view(M, W, 1).repeat(1, 1, W)  # (M, W, W)
    y = row_indices.view(M, 1, W).repeat(1, W, 1)  # (M, W, W)
    return torch.stack((x, y), dim=-1)  # (M, W, W, 2)
