import os
import numpy as np
import torch
import torch.nn.functional as F
import PIL.Image as Image

from .base_flow import BaseFlow

DEFAULT_IMAGE_PATH = "/home/yangdonglin/MotionCraft/src/flowsrepo/data/fire/fire_scene.png"
DEFAULT_MASK_PATH  = "/home/yangdonglin/MotionCraft/src/flowsrepo/data/fire/smoke_mask.png"

def _load_mask_png(mask_path: str, N: int) -> torch.Tensor:
    """return (1,1,N,N) float32 in [0,1]"""
    m = Image.open(mask_path).convert("L").resize((N, N))
    m = torch.from_numpy(np.array(m)).float() / 255.0
    return m[None, None, ...]


def _smooth_noise(noise: torch.Tensor, k: int = 9, iters: int = 2) -> torch.Tensor:
    """
    noise: (1,1,N,N)
    simple blur using avg_pool2d repeated
    """
    x = noise
    for _ in range(iters):
        x = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
    return x


def _dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """mask: (1,1,N,N) -> dilated mask (maxpool)"""
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius)


class SmokeFlow(BaseFlow):
    """
    SmokeFlow (MotionCraft):
    - 背景完全不动（mask外 identity）
    - 烟区域：上升 + 横向扰动 + 轻微扩散（由平滑噪声控制）
    - 不递推 mask（避免“腐蚀消失”），而是：mask_t = dilate(mask0, radius=t * grow_per_frame)

    你只要提供：
    - scene image
    - smoke mask
    """

    def __init__(
        self,
        N: int,
        image_path: str = None,
        mask_path: str = None,
        T: int = 48,
        drift_up_px: float = 1.2,
        sway_px: float = 0.8,
        curl_px: float = 0.6,
        noise_seed: int = 123,
        noise_smooth_k: int = 15,
        noise_smooth_iters: int = 2,
        grow_per_frame: float = 0.15,
    ):
        super().__init__(N=N)

        # ✅ 不传就用默认路径
        self.image_path = image_path or DEFAULT_IMAGE_PATH
        self.mask_path = mask_path or DEFAULT_MASK_PATH

        # （可选但强烈建议）启动时检查文件是否存在，方便排错
        assert os.path.exists(self.image_path), f"image not found: {self.image_path}"
        assert os.path.exists(self.mask_path),  f"mask not found: {self.mask_path}"

        self.T = int(T)

        self.drift_up_px = float(drift_up_px)
        self.sway_px = float(sway_px)
        self.curl_px = float(curl_px)

        self.noise_seed = int(noise_seed)
        self.noise_smooth_k = int(noise_smooth_k)
        self.noise_smooth_iters = int(noise_smooth_iters)
        self.grow_per_frame = float(grow_per_frame)

        # load base mask (1,1,N,N)
        self.mask0 = _load_mask_png(self.mask_path, self.N)

        # precompute per-frame grids
        self.precomputed_grids = self._precompute_grids()

        # prompts（给 latent pipeline 用；pixel 脚本不会用到）
        self.pos_prompt = "aerial view of a building fire with thick smoke, realistic, detailed"

    def get_default_image(self):
        return Image.open(self.image_path).convert("RGB")

    def get_default_framesteps(self):
        # clip_maker_pixel 会把 framestep 直接传给 get_flow(t)
        return list(range(self.T))

    def _mask_at(self, t: int) -> torch.Tensor:
        # mask 随时间轻微膨胀，让烟更“扩散”
        r = int(round(t * self.grow_per_frame))
        m = _dilate_mask(self.mask0, radius=r)
        return (m > 0.5).float()  # hard mask (1,1,N,N)

    def _precompute_grids(self):
        torch.manual_seed(self.noise_seed)

        N = self.N
        XY = self.XY.float()  # (N,N,2)

        grids = []

        # 固定一份“低频空间噪声基底”，每帧再做轻微变化（更像烟）
        base_noise = torch.randn(1, 1, N, N)
        base_noise = _smooth_noise(base_noise, k=self.noise_smooth_k, iters=self.noise_smooth_iters)
        base_noise = (base_noise - base_noise.min()) / (base_noise.max() - base_noise.min() + 1e-6)  # 0..1
        base_noise = base_noise * 2 - 1  # -1..1

        for t in range(self.T):
            mask_t = self._mask_at(t)  # (1,1,N,N)

            # 每帧再加一点点变化噪声（不要太强）
            frame_noise = torch.randn(1, 1, N, N) * 0.35
            frame_noise = _smooth_noise(frame_noise, k=self.noise_smooth_k, iters=self.noise_smooth_iters)
            frame_noise = (frame_noise - frame_noise.min()) / (frame_noise.max() - frame_noise.min() + 1e-6)
            frame_noise = frame_noise * 2 - 1

            n = 0.7 * base_noise + 0.3 * frame_noise  # -1..1

            # flow in pixel units (dx, dy) on N-grid
            # dy: 负号代表向上
            dy = (-self.drift_up_px) * torch.ones_like(n)

            # dx：横向摆动（随 y 稍微变化更自然）
            # 用 n 控制方向与大小
            dx = self.sway_px * n

            # “卷动/扩散”：用噪声的梯度近似（让烟边缘更活）
            # 简单差分求 gradient
            n_pad = F.pad(n, (1, 1, 1, 1), mode="reflect")
            gx = (n_pad[:, :, 1:-1, 2:] - n_pad[:, :, 1:-1, :-2]) * 0.5
            gy = (n_pad[:, :, 2:, 1:-1] - n_pad[:, :, :-2, 1:-1]) * 0.5
            dx = dx + self.curl_px * gx
            dy = dy + self.curl_px * gy

            flow = torch.cat([dx, dy], dim=1)  # (1,2,N,N)

            # ✅ 关键：mask 外 flow = 0 -> 背景不动
            flow = flow * mask_t

            # convert to sampling grid: coords = XY - flow
            # XY: (N,N,2), flow: (1,2,N,N)->(N,N,2)
            flow_xy = flow[0].permute(1, 2, 0)  # (N,N,2)
            coords = XY - flow_xy  # (N,N,2) pixel coords

            # normalize to [-1,1]
            grid = coords.clone()
            grid[..., 0] = 2.0 * (grid[..., 0] / (N - 1.0)) - 1.0
            grid[..., 1] = 2.0 * (grid[..., 1] / (N - 1.0)) - 1.0

            grids.append(grid)

        return grids

    def get_flow(self, t):
        t = int(t)
        t = max(0, min(t, self.T - 1))
        return self.precomputed_grids[t]
