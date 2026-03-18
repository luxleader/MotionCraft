import os
import numpy as np
import torch
import torch.nn.functional as F
import PIL.Image as Image
from .base_flow import BaseFlow

# ================== 路径（按你机器改） ==================
DEFAULT_IMAGE_PATH = "/home/yangdonglin/MotionCraft/src/flowsrepo/data/fire/fire_scene.png"
DEFAULT_FIRE_MASK_PATH = "/home/yangdonglin/MotionCraft/src/flowsrepo/data/fire/fire_mask.png"
DEFAULT_FUEL_MASK_PATH = "/home/yangdonglin/MotionCraft/src/flowsrepo/data/fire/fuel_mask.png"

# ================== 工具函数 ==================
def _load_mask_png(path: str, N: int):
    m = Image.open(path).convert("L").resize((N, N))
    m = torch.from_numpy(np.array(m)).float() / 255.0
    return (m > 0.5).float()[None, None, ...]  # (1,1,N,N)

def _dilate(x: torch.Tensor, r: int):
    if r <= 0: return x
    k = 2 * r + 1
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=r)

def _smooth_noise(x, k=9, iters=1):
    for _ in range(iters):
        x = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
    return x

def _coords_to_index(coords_xy: torch.Tensor, N: int) -> torch.Tensor:
    idx = coords_xy.round().long()
    idx[..., 0] = idx[..., 0].clamp(0, N - 1)
    idx[..., 1] = idx[..., 1].clamp(0, N - 1)
    return idx

# ================== FireFlow ==================
class FireFlow(BaseFlow):
    """
    Fire spreading flow for MotionCraft:
    - 加入了【环境风力】，引导火势向指定方向（右侧）强力蔓延
    - 移除了过度僵硬的固体锁，恢复火势流动性
    - 保留【极严背景保护】，火没烧到的地方绝对不产生变形拉扯
    """
    def __init__(
        self,
        N: int,
        image_path: str = None,
        fire_mask_path: str = None,
        fuel_mask_path: str = None,
        T: int = 48,

        # ---------- motion ----------
        drift_up_px: float = 2.0,    # 向上升腾的速度
        wind_x_px: float = 2.5,      # ✅ 核心新增：向右吹的风力（正数向右，负数向左），引导火往右边房子跑
        curl_px: float = 0.8,        # 翻卷扰动
        sway_px: float = 0.6,        # 横向摇摆
        expand_force: float = 1.0,   # 热膨胀力

        # ---------- spreading ----------
        spread_px_per_frame: int = 3,   # ✅ 核心修改：调快重绘区蔓延速度，确保能覆盖到右侧房屋
        spread_accel: float = 1.0,      
        fuel_dilate: int = 4,           

        # ---------- noise ----------
        noise_seed: int = 123,
        noise_smooth_k: int = 9,
        noise_smooth_iters: int = 1,
    ):
        super().__init__(N=N)

        self.image_path = image_path or DEFAULT_IMAGE_PATH
        self.fire_mask_path = fire_mask_path or DEFAULT_FIRE_MASK_PATH
        self.fuel_mask_path = fuel_mask_path or DEFAULT_FUEL_MASK_PATH
        self.T = T

        for p in [self.image_path, self.fire_mask_path, self.fuel_mask_path]:
            assert os.path.exists(p), f"Missing file: {p}"

        self.drift_up_px = drift_up_px
        self.wind_x_px = wind_x_px
        self.curl_px = curl_px
        self.sway_px = sway_px
        self.expand_force = float(expand_force)
        self.spread_px_per_frame = int(spread_px_per_frame)
        self.spread_accel = float(spread_accel)
        self.fuel_dilate = int(fuel_dilate)

        self.noise_seed = int(noise_seed)
        self.noise_smooth_k = int(noise_smooth_k)
        self.noise_smooth_iters = int(noise_smooth_iters)

        # load masks
        fire0 = _load_mask_png(self.fire_mask_path, N)
        fuel = _load_mask_png(self.fuel_mask_path, N)
        if self.fuel_dilate > 0:
            fuel = _dilate(fuel, self.fuel_dilate)

        self.fire0 = fire0
        self.fuel = fuel

        # precompute
        self.burn_seq = self._precompute_burn()
        self.precomputed_eta = self.burn_seq
        self.precomputed_grids = self._precompute_grids()

        self.pos_prompt = "aerial view of buildings on fire, flames spreading to the right, realistic fire"

    def get_default_image(self):
        return Image.open(self.image_path).convert("RGB")

    def get_default_framesteps(self):
        return list(range(self.T))

    def get_spatial_eta(self, t: int):
        t = max(0, min(int(t), self.T - 1))
        return self.precomputed_eta[t]

    def get_flow(self, t: int):
        t = max(0, min(int(t), self.T - 1))
        return self.precomputed_grids[t]

    def _frame_spread_rate(self, t: int) -> int:
        if self.T <= 1: return max(1, self.spread_px_per_frame)
        rate = self.spread_px_per_frame * (1.0 + self.spread_accel * (float(t) / float(self.T - 1)))
        return max(1, int(round(rate)))

    def _precompute_burn(self):
        """严格受 fuel 限制的布尔蔓延序列"""
        region = self.fire0.clone()
        burn_seq = []
        for t in range(self.T):
            rate = self._frame_spread_rate(t)
            grow = _dilate(region, rate)
            grow_fuel = grow * self.fuel
            region = torch.clamp(region + grow_fuel, 0.0, 1.0)
            burn_seq.append(region.clone())
        return burn_seq

    def _precompute_grids(self):
        torch.manual_seed(self.noise_seed)
        N = self.N
        XY = self.XY.float()
        grids = []

        base_noise = torch.randn(1, 1, N, N)
        base_noise = _smooth_noise(base_noise, self.noise_smooth_k, self.noise_smooth_iters)

        for t in range(self.T):
            n = base_noise

            # 1. 基础运动：卷曲 + 上升气流 + 【向右的环境风】
            n_pad = F.pad(n, (1, 1, 1, 1), mode="reflect")
            gx = (n_pad[:, :, 1:-1, 2:] - n_pad[:, :, 1:-1, :-2]) * 0.5
            gy = (n_pad[:, :, 2:, 1:-1] - n_pad[:, :, :-2, 1:-1]) * 0.5

            # dx 中叠加了 wind_x_px，强制将所有火苗向右推
            dx = self.sway_px * n + self.curl_px * gx + self.wind_x_px
            dy = -self.drift_up_px * torch.ones_like(n) + self.curl_px * gy
            
            dx = dx[0, 0] 
            dy = dy[0, 0]

            # 2. 密度梯度膨胀力（向外扩散）
            burn = self.burn_seq[t]
            density = burn.float().clone()
            for _ in range(5):
                density = F.avg_pool2d(density, kernel_size=11, stride=1, padding=5)
            
            grad_y, grad_x = torch.gradient(density[0, 0])
            
            k_expand = self.expand_force * 30.0 
            dx = dx - k_expand * grad_x.to(dx.device)
            dy = dy - k_expand * grad_y.to(dy.device)

            # 3. 运动限制：不在火场附近的地方保持绝对静止
            adv_mask = _dilate(burn, 5)[0, 0].to(XY.device)
            dx = dx * adv_mask
            dy = dy * adv_mask

            flow = torch.stack([dx, dy], dim=-1) # (N, N, 2)
            coords = XY - flow

            # ================== ✅ 核心机制：极严背景保护 (Source Guard) ==================
            # 作用：只要火还没烧到右侧房屋，房屋绝对不会被吸进去变形。
            # 一旦 burn_seq 蔓延到了右侧房屋，该区域解锁，房屋瞬间爆燃并化作向右上方飞散的火苗。
            source_mask = _dilate(burn, 1)[0, 0].to(XY.device)
            idx = _coords_to_index(coords, N)
            
            valid = source_mask[idx[..., 1], idx[..., 0]]  
            valid = (valid > 0.5).to(torch.float32)[..., None]  
            
            coords = coords * valid + XY * (1.0 - valid)
            # =================================================================================

            # 4. 归一化输出
            grid = coords.clone()
            grid[..., 0] = 2 * (grid[..., 0] / (N - 1)) - 1
            grid[..., 1] = 2 * (grid[..., 1] / (N - 1)) - 1
            grids.append(grid)

        return grids