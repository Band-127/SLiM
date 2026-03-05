import os
import torch
import numpy as np
from torch.utils.data import Dataset
from skimage.io import imread


class AachenDayNightEvalDataset(Dataset):
    def __init__(self, root, pair_list, image_size=1600):
        self.root = root
        self.image_size = image_size

        # 读取图像对列表
        with open(pair_list, "r") as f:
            self.pairs = [line.strip().split() for line in f.readlines()]

    def __len__(self):
        return len(self.pairs)

    def load_process_image(self, path, output_size=None):
        # 加载图像并转换为tensor
        img = imread(os.path.join(self.root, path)).astype(np.float32) / 256.0
        tensor = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
        tensor = tensor.mean(dim=0, keepdim=True)  # 灰度化 [1, H, W]

        h, w = tensor.shape[1], tensor.shape[2]

        if output_size is None:
            # 控制长边到指定尺寸
            long_side = max(h, w)
            scale = self.image_size / long_side
            new_h, new_w = int(h * scale), int(w * scale)

            # 调整长边为32的倍数
            if new_h > new_w:
                new_h = (new_h + 31) // 32 * 32
            else:
                new_w = (new_w + 31) // 32 * 32
        else:
            new_h, new_w = output_size

        # 缩放处理
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0).float(),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # 计算缩放因子 [ori_w/new_w, ori_h/new_h]
        scale_w = w / new_w
        scale_h = h / new_h
        return tensor, torch.tensor([scale_w, scale_h]), (new_h, new_w)

    def __getitem__(self, idx):
        query_path, ref_path = self.pairs[idx]

        # 先处理query图像获取目标尺寸
        image0, scale0, output_size = self.load_process_image(query_path)
        # 强制reference图像使用相同尺寸
        image1, scale1, _ = self.load_process_image(ref_path, output_size)

        # 解析元数据
        scene_id = query_path.split("/")[1]  # 假设路径格式为"day/场景/图像"
        data_type = "day" if "day" in query_path else "night"

        return {
            "image0": image0,  # [1, H, W]
            "image1": image1,  # [1, H, W]
            "scene_id": scene_id,
            "data_type": data_type,
            "scale0": scale0,
            "scale1": scale1,
        }
