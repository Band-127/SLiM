import torch
import torch.nn.functional as F


@torch.no_grad
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
    """Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    Modified code from https://github.com/zju3dv/LoFTR, add grid_sample to get depth from continuous points.

    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    # Get depth using grid sample from torch
    h, w = depth0.shape[1:]
    kpts0_normalized = kpts0.clone()
    kpts0_normalized[:, :, 0] = (kpts0[:, :, 0] / (w)) * 2 - 1
    kpts0_normalized[:, :, 1] = (kpts0[:, :, 1] / (h)) * 2 - 1

    kpts0_normalized = kpts0_normalized.unsqueeze(1)  # (N, 1, L, 2)
    depth0 = depth0.unsqueeze(1)  # Add depth channel (N, 1, H, W)
    kpts0_depth = F.grid_sample(
        depth0,
        kpts0_normalized,
        mode="bilinear",
        align_corners=False,
    )
    kpts0_depth = kpts0_depth.squeeze(1).squeeze(1)  # (N, L)
    # The rest code are same as in LoFTR project
    nonzero_mask = kpts0_depth != 0

    # Unproject
    kpts0_h = (
        torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1)
        * kpts0_depth[..., None]
    )  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (
        w_kpts0_h[:, :, [2]] + 1e-4
    )  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (
        (w_kpts0[:, :, 0] > 0)
        * (w_kpts0[:, :, 0] < w - 1)
        * (w_kpts0[:, :, 1] > 0)
        * (w_kpts0[:, :, 1] < h - 1)
    )
    w_kpts0_long = w_kpts0.long()
    w_kpts0_long[~covisible_mask, :] = 0

    w_kpts0_depth = torch.stack(
        [
            depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]]
            for i in range(w_kpts0_long.shape[0])
        ],
        dim=0,
    )  # (N, L)
    consistent_mask = (
        (w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth
    ).abs() < 0.2
    valid_mask = nonzero_mask * covisible_mask * consistent_mask

    return valid_mask, w_kpts0
