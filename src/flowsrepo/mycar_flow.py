import torch
import PIL.Image as Image
import numpy as np
import os
from .base_flow import BaseFlow


class MyCarFlow(BaseFlow):
    def __init__(self, N) -> None:
        super().__init__(N=N)
        self.this_path = os.path.dirname(os.path.abspath(__file__))

        # =========================
        # 可调参数
        # =========================
        # 控制整体运动幅度（一般 1.0 即可）
        self.flow_scale = 1.0

        # =========================
        # 1. 加载质心光流 (T,2,H,W)
        # =========================
        flow_hist = self.load_flow_data()

        # =========================
        # 2. 加载初始 mask
        # =========================
        current_mask = self.load_mask_data().to(self.XY.device)

        # =========================
        # 3. 初始化缓存
        # =========================
        self.precomputed_bin_masks = [current_mask]

        # 第 0 帧：恒等采样网格
        identity_grid = (self.XY.float() / (self.N - 1)) * 2 - 1
        self.precomputed_flows = [identity_grid]

        # =========================
        # 4. 逐帧推进 mask & flow
        # =========================
        for t in range(flow_hist.shape[0]):
            # flow_pixels: (1, H, W, 2)
            flow_pixels = flow_hist[[t]].to(self.XY.device) * self.flow_scale

            # --- 核心逻辑 ---
            # grid_sample 是 backward warp：
            # 要找“从哪里采样”，所以是 current - flow
            sampling_coords_pixel = self.XY - flow_pixels

            # 像素坐标 -> [-1,1]
            sampling_grid = 2.0 * sampling_coords_pixel / (self.N - 1) - 1.0
            sampling_grid = sampling_grid.to(torch.float32)

            # warp mask
            current_mask = torch.nn.functional.grid_sample(
                current_mask,
                sampling_grid,
                align_corners=True,
                mode="bilinear",
                padding_mode="zeros",
            )

            # 二值化，保证 mask 干净
            current_mask = (current_mask > 0.5).float()

            self.precomputed_flows.append(sampling_grid.squeeze(0))
            self.precomputed_bin_masks.append(current_mask)

        # =========================
        # 5. 文本提示
        # =========================
        self.pos_prompt = "A car is moving down the road from top to bottom, while the rest of the background remains still, high quality, photorealistic, 4k"

    # ==========================================================
    # 接口函数（BaseFlow 要求）
    # ==========================================================

    def get_spatial_eta(self, t):
        if t < len(self.precomputed_bin_masks):
            return self.precomputed_bin_masks[t]
        return self.precomputed_bin_masks[-1]

    def get_flow(self, t) -> torch.Tensor:
        if t < len(self.precomputed_flows):
            return self.precomputed_flows[t]
        return self.precomputed_flows[-1]

    def get_default_framesteps(self) -> torch.Tensor:
        return torch.tensor(list(range(len(self.precomputed_flows))))

    def get_default_image(self) -> torch.Tensor:
        bg_path = f"{self.this_path}/data/mycar/mycar_512.png"
        if not os.path.exists(bg_path):
            print(f"Warning: {bg_path} not found.")
        return Image.open(bg_path).convert("RGB")

    # ==========================================================
    # 数据加载
    # ==========================================================

    def load_mask_data(self):
        """
        mycar_mask_std.npy : (T,1,H,W)
        这里只取第 0 帧作为初始位置
        """
        path = f"{self.this_path}/data/mycar/mycar_mask_std.npy"
        mask = np.load(path)
        mask = torch.from_numpy(mask).float()

        init_mask = mask[0:1]  # (1,1,H,W)

        init_mask = torch.nn.functional.interpolate(
            init_mask, size=[self.N, self.N], mode="nearest"
        )
        return init_mask

    def load_flow_data(self):
        """
        mycar_flow_centroid.npy : (T,2,H,W)
        表示像素位移（dx,dy）
        """
        path = f"{self.this_path}/data/mycar/mycar_flow_centroid.npy"
        flow = np.load(path)
        flow = torch.from_numpy(flow).float()  # (T,2,H,W)

        T, _, H, W = flow.shape

        # 插值到目标分辨率
        flow = torch.nn.functional.interpolate(
            flow, size=[self.N, self.N], mode="bilinear", align_corners=True
        )

        # 像素尺度修正
        scale_ratio = self.N / float(H)
        flow = flow * scale_ratio

        # (T,2,H,W) -> (T,H,W,2)
        flow = flow.permute(0, 2, 3, 1)

        return flow
