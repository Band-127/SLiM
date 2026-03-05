from loguru import logger
import torch
from einops import repeat
from kornia.utils import create_meshgrid
from einops.einops import rearrange

from .misc import create_grid
from .geometry import warp_kpts


##############  ↓  Coarse-Level supervision  ↓  ##############
@torch.no_grad
def mask_pts_at_padded_regions(grid_pt, mask):
    """
    将填充区域的点设置为零, 用于处理megadepth数据集中的零填充图像。

    Args:
        grid_pt (torch.Tensor): 输入的网格点, 形状为[N, hw, 2]。
        mask (torch.Tensor): 掩码, 形状为[N, H, W], 指示有效区域。

    Returns:
        torch.Tensor: 处理后的网格点, 填充区域的点被设置为零。
    """
    # For megadepth dataset, zero-padding exists in images
    mask = repeat(mask, "n h w -> n (h w) c", c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt


@torch.no_grad
def spvs_coarse(data, coarse_scale):
    """
    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }

    NOTE:
        - for scannet dataset, there're 3 kinds of resolution {i, c, f}
        - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """
    # 1. misc
    device = data["image0"].device
    # 获取图像所在的设备
    N, _, H0, W0 = data["image0"].shape
    # 获取第一张图像的形状
    _, _, H1, W1 = data["image1"].shape
    # 获取第二张图像的形状
    scale = coarse_scale
    scale0 = (
        scale * data["scale0"][:, None]
        if "scale0" in data
        else torch.tensor([[scale, scale]], device=device).repeat(N, 1)[:, None]
    )
    # 计算尺度0
    scale1 = (
        scale * data["scale1"][:, None]
        if "scale1" in data
        else torch.tensor([[scale, scale]], device=device).repeat(N, 1)[:, None]
    )
    # 计算尺度1
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])
    # 计算缩放后的高宽

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    # 在网格中创建关键点并调整为图像分辨率
    grid_pt0_c = (
        create_meshgrid(h0, w0, False, device).reshape(1, h0 * w0, 2).repeat(N, 1, 1)
    ) + 0.5  # [N, hw, 2]
    grid_pt0_i = scale0 * grid_pt0_c
    grid_pt1_c = (
        create_meshgrid(h1, w1, False, device).reshape(1, h1 * w1, 2).repeat(N, 1, 1)
    ) + 0.5  # [N, hw, 2]
    grid_pt1_i = scale1 * grid_pt1_c

    # mask padded region to (0, 0), so no need to manually mask conf_matrix_gt
    # 将填充区域的网格点设置为(0, 0), 以避免手动掩码conf_matrix_gt
    if "mask0" in data:
        grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data["mask0"])
        grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data["mask1"])

    # warp kpts bi-directionally and resize them to coarse-level resolution
    # 双向扭曲关键点并调整为粗级别分辨率
    # (no depth consistency check, since it leads to worse results experimentally)
    # (unhandled edge case: points with 0-depth will be warped to the left-up corner)
    _, w_pt0_i = warp_kpts(
        grid_pt0_i,
        data["depth0"],
        data["depth1"],
        data["T_0to1"],
        data["K0"],
        data["K1"],
    )
    _, w_pt1_i = warp_kpts(
        grid_pt1_i,
        data["depth1"],
        data["depth0"],
        data["T_1to0"],
        data["K1"],
        data["K0"],
    )
    w_pt0_c = w_pt0_i / scale1 - 0.5
    w_pt1_c = w_pt1_i / scale0 - 0.5

    # 3. check if mutual nearest neighbor
    w_pt0_c_round = w_pt0_c[:, :, :].round().long()
    nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1
    w_pt1_c_round = w_pt1_c[:, :, :].round().long()
    nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0

    # corner case: out of boundary
    # 边界情况处理：超出边界的点
    def out_bound_mask(pt, w, h):
        return (
            (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
        )

    nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0
    nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0)] = 0

    loop_back = torch.stack(
        [nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], dim=0
    )
    correct_0to1 = loop_back == torch.arange(h0 * w0, device=device)[None].repeat(N, 1)
    correct_0to1[:, 0] = False  # ignore the top-left corner
    # 忽略左上角

    # 4. construct a gt conf_matrix
    conf_matrix_gt = torch.zeros(N, h0 * w0, h1 * w1, device=device)
    b_ids, i_ids = torch.where(correct_0to1 != 0)
    j_ids = nearest_index1[b_ids, i_ids]

    conf_matrix_gt[b_ids, i_ids, j_ids] = 1
    data.update({"conf_matrix_gt": conf_matrix_gt})

    # 5. save coarse matches(gt) for training fine level
    if len(b_ids) == 0:
        logger.warning(f"No groundtruth coarse match found for: {data['pair_names']}")
        # this won't affect fine-level loss calculation
        # 这不会影响细级别损失计算
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)

    data.update({"spv_b_ids": b_ids, "spv_i_ids": i_ids, "spv_j_ids": j_ids})


def compute_supervision_coarse(data, coarse_scale, config):
    assert len(set(data["dataset_name"])) == 1, (
        "Do not support mixed datasets training!"
    )
    data_source = data["dataset_name"][0]
    if data_source.lower() in ["scannet", "megadepth"]:
        spvs_coarse(data, coarse_scale)
        spvs_intermediate(data, config)
    else:
        raise ValueError(f"Unknown data source: {data_source}")


##############  ↓  Intermediate-Level supervision  ↓  ##############
def get_coarse_coord(data, config):
    b_idx_c = data["spv_b_ids"]
    i_idx_c = data["spv_i_ids"]
    j_idx_c = data["spv_j_ids"]
    coarse_scale = config["MODEL"]["COARSE_SCALE"]
    W0_c = data["image0"].shape[3] // coarse_scale
    W1_c = data["image1"].shape[3] // coarse_scale

    device = data["image0"].device
    N = data["image0"].shape[0]

    # Match indices -> coordinates
    scale0 = (
        coarse_scale * data["scale0"][b_idx_c]
        if "scale0" in data
        else torch.tensor([[coarse_scale, coarse_scale]], device=device).repeat(N, 1)[
            b_idx_c
        ]
    )
    scale1 = (
        coarse_scale * data["scale1"][b_idx_c]
        if "scale1" in data
        else torch.tensor([[coarse_scale, coarse_scale]], device=device).repeat(N, 1)[
            b_idx_c
        ]
    )
    coarse_coord_0 = (
        torch.stack(
            (
                (i_idx_c % W0_c),
                (i_idx_c // W0_c),
            ),
            dim=1,
        )
        * scale0
    )
    coarse_coord_1 = (
        torch.stack(
            (
                (j_idx_c % W1_c),
                (j_idx_c // W1_c),
            ),
            dim=1,
        )
        * scale1
    )
    return coarse_coord_0, coarse_coord_1


@torch.no_grad
def spvs_intermediate(data, config):
    device = data["image0"].device
    N, _, H0, W0 = data["image0"].shape
    _, _, H1, W1 = data["image1"].shape
    coarse_coord_0, coarse_coord_1 = get_coarse_coord(data, config)
    coarse_scale = config["MODEL"]["COARSE_SCALE"]
    # EDITED!!!
    # fine_scale = config["MODEL"]["FINE_SCALE"]
    fine_scale = 1
    # EDITED!!!
    b_idx_c = data["spv_b_ids"]  # [M]

    window_size = int(coarse_scale / fine_scale)  # w
    absolute_fine_scale0 = (
        fine_scale * data["scale0"][b_idx_c]
        if "scale0" in data
        else torch.tensor([[fine_scale, fine_scale]], device=device).repeat(N, 1)[
            b_idx_c
        ]
    )  # [M, 2]
    absolute_fine_scale1 = (
        fine_scale * data["scale1"][b_idx_c]
        if "scale1" in data
        else torch.tensor([[fine_scale, fine_scale]], device=device).repeat(N, 1)[
            b_idx_c
        ]
    )  # [M, 2]
    # Indices for the top-left corner of the window
    intermediate_coord_0 = (
        coarse_coord_0 / absolute_fine_scale0
    ).round() + 0.5  # [M, 2]
    intermediate_coord_1 = (
        coarse_coord_1 / absolute_fine_scale1
    ).round() + 0.5  # [M, 2]

    b_idx_c_unique = b_idx_c.unique(sorted=True)  # [B]
    offset = torch.arange(window_size, device=device, dtype=torch.long).unsqueeze(
        0
    )  # [1, w]
    conf_matrix_gt = torch.zeros(
        b_idx_c.shape[0], window_size**2, window_size**2, device=device
    )  # [M, h0*w0, h1*w1]

    def out_bound_mask(pt, w, h):
        return (
            (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
        )

    b_ids_all = []
    m_ids_all = []
    i_ids_all = []
    j_ids_all = []
    intermediate_coord_0_all = []
    intermediate_coord_1_all = []
    # Calculate for points per image batch
    for b in b_idx_c_unique:
        batch_mask = b_idx_c == b  # [M]
        absolute_fine_scale0 = (
            fine_scale * data["scale0"][b]
            if "scale0" in data
            else torch.tensor([fine_scale, fine_scale], device=device)
        )  # [2]
        absolute_fine_scale1 = (
            fine_scale * data["scale1"][b]
            if "scale1" in data
            else torch.tensor([fine_scale, fine_scale], device=device)
        )  # [2]
        intermediate_coord_0_single_batch = intermediate_coord_0[batch_mask]  # [M', 2]
        intermediate_coord_1_single_batch = intermediate_coord_1[batch_mask]  # [M', 2]

        # Create coord grid for pairs in the window of image0 and image1
        row_coord_0 = (
            intermediate_coord_0_single_batch[:, 1].unsqueeze(1) + offset
        )  # [M', w]
        col_coord_0 = (
            intermediate_coord_0_single_batch[:, 0].unsqueeze(1) + offset
        )  # [M', w]
        coord_grid_0 = (
            create_grid(row_coord_0, col_coord_0).permute(0, 2, 1, 3).float()
        )  # [M', w, w, 2], (x, y)
        row_coord_1 = (
            intermediate_coord_1_single_batch[:, 1].unsqueeze(1) + offset
        )  # [M', w]
        col_coord_1 = (
            intermediate_coord_1_single_batch[:, 0].unsqueeze(1) + offset
        )  # [M', w]
        coord_grid_1 = (
            create_grid(row_coord_1, col_coord_1).permute(0, 2, 1, 3).float()
        )  # [M', w, w, 2], (x, y)

        # To absolute coordinate
        coord_grid_0 *= absolute_fine_scale0.unsqueeze(0).unsqueeze(0)  # [M', w, w, 2]
        coord_grid_1 *= absolute_fine_scale1.unsqueeze(0).unsqueeze(0)  # [M', w, w, 2]
        coord_grid_0 = rearrange(
            coord_grid_0, "m h w c -> 1 (m h w) c"
        )  # [1，M'*w*w, 2]
        coord_grid_1 = rearrange(
            coord_grid_1, "m h w c -> 1 (m h w) c"
        )  # [1，M'*w*w, 2]

        # Warp
        _, w_pt0_i = warp_kpts(
            coord_grid_0,
            data["depth0"][b].unsqueeze(0),
            data["depth1"][b].unsqueeze(0),
            data["T_0to1"][b].unsqueeze(0),
            data["K0"][b].unsqueeze(0),
            data["K1"][b].unsqueeze(0),
        )
        _, w_pt1_i = warp_kpts(
            coord_grid_1,
            data["depth1"][b].unsqueeze(0),
            data["depth0"][b].unsqueeze(0),
            data["T_1to0"][b].unsqueeze(0),
            data["K1"][b].unsqueeze(0),
            data["K0"][b].unsqueeze(0),
        )
        w_pt0_i = rearrange(
            w_pt0_i,
            "b (m h w) c ->(b m) (h w) c",
            b=1,
            m=intermediate_coord_0_single_batch.shape[0],
            h=window_size,
            w=window_size,
            c=2,
        )  # [M', h0 * w0, 2]
        w_pt1_i = rearrange(
            w_pt1_i,
            "b (m h w) c ->(b m) (h w) c",
            b=1,
            m=intermediate_coord_1_single_batch.shape[0],
            h=window_size,
            w=window_size,
            c=2,
        )  # [M', h1 * w1, 2]

        # Back to relative coordinate
        w_pt0_i /= absolute_fine_scale1.unsqueeze(0).unsqueeze(0)  # [M', h0 * w0, 2]
        w_pt1_i /= absolute_fine_scale0.unsqueeze(0).unsqueeze(0)  # [M', h1 * w1, 2]
        # Convert to indices of window by subtracting the coord of top-left corner
        w_pt0_i_round = (
            (w_pt0_i - intermediate_coord_1_single_batch.unsqueeze(1)).round().long()
        )  # [M', h0 * w0, 2]
        w_pt1_i_round = (
            (w_pt1_i - intermediate_coord_0_single_batch.unsqueeze(1)).round().long()
        )  # [M', h1 * w1, 2]
        # Get the nearest warped index in the window
        nearest_index1 = w_pt0_i_round[..., 0] + w_pt0_i_round[..., 1] * (window_size)
        nearest_index0 = w_pt1_i_round[..., 0] + w_pt1_i_round[..., 1] * (window_size)

        # corner case: out of boundary
        nearest_index1[out_bound_mask(w_pt0_i_round, window_size, window_size)] = 0
        nearest_index0[out_bound_mask(w_pt1_i_round, window_size, window_size)] = 0

        # Construct conf_matrix_gt[M', h0*w0, h1*w1]
        # loop_back = torch.stack(
        #     [nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], dim=0
        # )
        batch_indices = torch.arange(
            nearest_index1.size(0), device=nearest_index1.device
        )
        loop_back = nearest_index0[batch_indices.unsqueeze(1), nearest_index1]
        correct_0to1 = loop_back == torch.arange(window_size**2, device=device)[
            None
        ].repeat(intermediate_coord_0_single_batch.shape[0], 1)
        b_ids, i_ids = torch.where(correct_0to1 != 0)
        j_ids = nearest_index1[b_ids, i_ids]
        conf_matrix_gt[b_ids, i_ids, j_ids] = 1

        # Record for each batch
        b_ids_all.append(b_idx_c[b_ids])
        m_ids_all.append(b_ids)
        i_ids_all.append(i_ids)
        j_ids_all.append(j_ids)
        coord_grid_0 = rearrange(
            coord_grid_0,
            "b (m h w) c -> (b m) (h w) c",
            b=1,
            m=intermediate_coord_1_single_batch.shape[0],
            h=window_size,
            w=window_size,
            c=2,
        )  # [1，M'*w*w, 2]
        coord_grid_1 = rearrange(
            coord_grid_1,
            "b (m h w) c -> (b m) (h w) c",
            b=1,
            m=intermediate_coord_0_single_batch.shape[0],
            h=window_size,
            w=window_size,
            c=2,
        )  # [1，M'*w*w, 2]
        intermediate_coord_0_all.append(coord_grid_0[b_ids, i_ids])
        intermediate_coord_1_all.append(coord_grid_1[b_ids, j_ids])
    b_ids_all = torch.cat(b_ids_all, dim=0)
    m_ids_all = torch.cat(m_ids_all, dim=0)
    i_ids_all = torch.cat(i_ids_all, dim=0)
    j_ids_all = torch.cat(j_ids_all, dim=0)
    intermediate_coord_0_all = torch.cat(intermediate_coord_0_all, dim=0)
    intermediate_coord_1_all = torch.cat(intermediate_coord_1_all, dim=0)

    data.update({"conf_matrix_f_gt": conf_matrix_gt})
    data.update(
        {
            "spv_b_ids_it": b_ids_all,
            "spv_m_ids_it": m_ids_all,
            "spv_i_ids_it": i_ids_all,
            "spv_j_ids_it": j_ids_all,
        }
    )
    data.update({"intermediate_coord_0_gt": intermediate_coord_0_all})
    data.update({"intermediate_coord_1_gt": intermediate_coord_1_all})


##############  ↓  Fine-Level supervision  ↓  ##############
@torch.no_grad
def spvs_fine(data, config):
    """
    Update:
        data (dict):{
            "coord_offset_f_gt": [N, 2],
            "correct_mask_1": [N],
        }
    """
    coarse_scale = config["MODEL"]["COARSE_SCALE"]
    # EDITED!!!
    # fine_scale = config["MODEL"]["FINE_SCALE"]
    fine_scale = 1
    # EDITED!!!
    device = data["image0"].device
    N = data["image0"].shape[0]
    absolute_scale1 = (
        fine_scale * data["scale1"]
        if "scale1" in data
        else torch.tensor([[fine_scale, fine_scale]], device=device).repeat(N, 1)
    )

    # Get predicted fine coordinates of image0 and batch_indices
    fine_coord_0 = data["fine_coord_0"]  # [N, 2]
    intermediate_coord_1 = data["intermediate_coord_1"]  # [N, 2]
    target_fine_coord_1 = torch.zeros_like(fine_coord_0)  # [N, 2]
    offset_f_1 = torch.zeros_like(fine_coord_0)  # [N, 2]
    radius = int(config["MODEL"]["REFINE_LOOKUP_RADIUS"])
    batch_idx = data["b_idx_it"]  # [N]
    # Warp all fine coordinates of image0 on image1
    sorted_batch_unique = batch_idx.unique(sorted=True)
    for b in sorted_batch_unique:
        batch_mask = batch_idx == b
        # Offset on image 1 based on fine coord of image 0
        fine_coord_0_single_batch = fine_coord_0[batch_mask].unsqueeze(0)
        _, target_fine_coord_1_single_batch = warp_kpts(
            fine_coord_0_single_batch,
            data["depth0"][b].unsqueeze(0),
            data["depth1"][b].unsqueeze(0),
            data["T_0to1"][b].unsqueeze(0),
            data["K0"][b].unsqueeze(0),
            data["K1"][b].unsqueeze(0),
        )
        offset_f_1[batch_mask] = (
            (target_fine_coord_1_single_batch[0] - intermediate_coord_1[batch_mask])
            / absolute_scale1[b]
            / radius
        )
        target_fine_coord_1[batch_mask] = target_fine_coord_1_single_batch[0]

    # Get correct mask
    correct_mask = (
        torch.linalg.norm(offset_f_1, ord=float("inf"), dim=1)
        < ((coarse_scale / fine_scale) / 2 / radius) * 1.5
    )

    data.update({"correct_mask": correct_mask, "coord_offset_gt": offset_f_1})
    data.update({"fine_coord_1_gt": target_fine_coord_1})


def compute_supervision_fine(data, config):
    data_source = data["dataset_name"][0]
    if data_source.lower() in ["scannet", "megadepth"]:
        spvs_fine(data, config)
    else:
        raise NotImplementedError
