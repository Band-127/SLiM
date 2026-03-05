import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from skimage.io import imread


class HPatchesEvalDataset(Dataset):
    def __init__(self, root, sequence_list, image_size=1600):
        self.root = root
        self.image_size = image_size

        # Read and process sequence list
        with open(sequence_list, "r") as f:
            all_lines = [line.strip() for line in f.readlines()]

        # Group into sequences of 6 images each
        self.sequences = []
        for i in range(0, len(all_lines), 6):
            images_list = all_lines[i : i + 6]
            images_list.sort()
            self.sequences.append(images_list)

    def __len__(self):
        return len(self.sequences) * 5  # 5 pairs per sequence

    def load_process_image(self, path, output_size=None):
        # Load image and convert to tensor
        img = imread(path).astype(np.float32) / 256.0
        tensor = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW

        # Convert to grayscale by averaging channels
        tensor = tensor.mean(dim=0, keepdim=True)  # [1, H, W]

        # Get original dimensions
        h, w = tensor.shape[1], tensor.shape[2]

        if output_size is None:
            # Calculate scaling factor for shortest edge
            if self.image_size is not None:
                scale = self.image_size / min(h, w)
                new_h, new_w = round(h * scale), round(w * scale)

                # Adjust longest edge to be multiple of 32
                if new_h > new_w:
                    new_h = int(np.ceil(new_h / 32)) * 32
                else:
                    new_w = int(np.ceil(new_w / 32)) * 32
            else:
                new_h = int(np.ceil(h / 32)) * 32
                new_w = int(np.ceil(w / 32)) * 32

        else:
            new_h, new_w = output_size

        # Resize with bilinear interpolation
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0).float(),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Calculate scale factors [ori_w/new_w, ori_h/new_h]
        scale_w = w / new_w
        scale_h = h / new_h
        scale_tensor = torch.tensor([scale_w, scale_h], dtype=torch.float32)

        return tensor, scale_tensor, torch.tensor([h, w]), (new_h, new_w)

    def __getitem__(self, idx):
        # Calculate sequence and pair indices
        seq_idx = idx // 5
        pair_idx = (idx % 5) + 1  # pairs 1-5

        seq_paths = self.sequences[seq_idx]
        query_path = os.path.join(self.root, seq_paths[0])
        ref_path = os.path.join(self.root, seq_paths[pair_idx])

        # Load source (image0) and target (image1)
        image0, scale0, ori_size_0, output_size0 = self.load_process_image(query_path)
        image1, scale1, ori_size_1, _ = self.load_process_image(ref_path, output_size0)

        # Parse metadata
        scene_dir = os.path.dirname(seq_paths[0])
        scene_id = os.path.basename(scene_dir)
        scene_type = "viewpoint" if scene_id.startswith("v_") else "illumination"

        # Load homography matrix (from source to target)
        H_path = os.path.join(self.root, scene_dir, f"H_1_{pair_idx + 1}")
        H = np.loadtxt(H_path)

        # Source image corners (original size)
        source_corners = np.array(
            [
                [0, 0],  # Top-left
                [ori_size_0[1] - 1, 0],  # Top-right
                [0, ori_size_0[0] - 1],  # Bottom-left
                [ori_size_0[1] - 1, ori_size_0[0] - 1],  # Bottom-right
            ],
            dtype=np.float32,
        )

        # Project corners to target space
        projected = cv2.perspectiveTransform(
            source_corners.reshape(1, -1, 2), H
        ).squeeze(0)

        return {
            "image0": image0,  # Source image [1, H, W]
            "image1": image1,  # Target image [1, H, W]
            "scene_id": scene_id,
            "scene_type": scene_type,
            "projection_coords": torch.from_numpy(projected).float(),  # [4, 2]
            "scale0": scale0,  # [ori_w/new_w, ori_h/new_h]
            "scale1": scale1,  # [ori_w/new_w, ori_h/new_h]
            "ori_size0": ori_size_0,
            "ori_size1": ori_size_1,
        }
