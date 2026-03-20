import os
import numpy as np
import torch
import torch.nn.functional as F
import PIL.Image as Image
from .base_flow import BaseFlow

DEFAULT_IMAGE_PATH = "/home/yangdonglin/MotionCraft/src/flowsrepo/data/water/water_scene.png"
DEFAULT_WATER_MASK_PATH = "/home/yangdonglin/MotionCraft/src/flowsrepo/data/water/water_1.png"
DEFAULT_REGION_MASK_PATH = "/home/yangdonglin/MotionCraft/src/flowsrepo/data/water/water_region_1.png"
DEFAULT_SOLID_MASK_PATH = "/home/yangdonglin/MotionCraft/src/flowsrepo/data/water/solid_mask.png"


def _load_mask_png(path: str, N: int):
    m = Image.open(path).convert("L").resize((N, N))
    m = torch.from_numpy(np.array(m)).float() / 255.0
    return (m > 0.5).float()[None, None, ...]


def _dilate(x: torch.Tensor, r: int):
    if r <= 0:
        return x
    k = 2 * r + 1
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=r)


def _smooth_noise(x, k=9, iters=1):
    for _ in range(iters):
        x = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
    return x


def _compute_mask_centroid_xy(mask_01: torch.Tensor):
    m = (mask_01[0, 0] > 0.5).float()
    ys, xs = torch.where(m > 0.5)
    if xs.numel() == 0:
        N = mask_01.shape[-1]
        return (0.5 * (N - 1), 0.5 * (N - 1))
    return xs.float().mean().item(), ys.float().mean().item()


def _radial_dist_map(N: int, cx: float, cy: float, device):
    yy, xx = torch.meshgrid(
        torch.arange(N, device=device, dtype=torch.float32),
        torch.arange(N, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.sqrt((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2)


def _xy_grid(N: int, device):
    yy, xx = torch.meshgrid(
        torch.arange(N, device=device, dtype=torch.float32),
        torch.arange(N, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return xx, yy


def _smoothstep01(x: torch.Tensor):
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _soft_eta_from_dist(dist_2d, level, band, gamma=1.0, core_boost=0.0):
    band = max(1e-6, float(band))
    x = (float(level) - dist_2d) / band + 0.5
    x = _smoothstep01(x)
    if gamma != 1.0:
        x = x.pow(float(gamma))
    if core_boost > 0.0:
        x = (x + core_boost * x * x).clamp(0.0, 1.0)
    return x


def _make_transition_ring(dist_2d, level, ring_inner_ratio, ring_outer_ratio, band):
    inner = level - ring_inner_ratio * band
    outer = level + ring_outer_ratio * band

    left = (dist_2d - inner) / max(1e-6, (level - inner))
    left = _smoothstep01(left)

    right = 1.0 - (dist_2d - level) / max(1e-6, (outer - level))
    right = _smoothstep01(right)

    return torch.minimum(left, right).clamp(0.0, 1.0)


def _partial_zero_mean_flow(dx: torch.Tensor, dy: torch.Tensor, weight: torch.Tensor, keep_ratio: float = 0.65):
    """
    不再完全去掉整体平移，只去掉一部分。
    keep_ratio 越大，保留的右下净漂移越强。
    """
    w = weight.clamp(0.0, 1.0)
    denom = w.sum().clamp_min(1e-6)
    mx = (dx * w).sum() / denom
    my = (dy * w).sum() / denom
    dx = dx - (1.0 - keep_ratio) * mx
    dy = dy - (1.0 - keep_ratio) * my
    return dx, dy


class FloodFlow(BaseFlow):
    def __init__(
        self,
        N: int,
        image_path: str = None,
        water_mask_path: str = None,
        region_mask_path: str = None,
        solid_mask_path: str = None,
        T: int = 48,

        # flooding
        level_start: float = 20.0,
        level_end: float = 200.0,      # 改回适合 128x128 尺寸的正常最大半径
        level_ease_pow: float = 1.2,
        eta_band: float = 15.0,        # 减小渐变宽度，让洪水边缘更实、更明显
        eta_gamma: float = 1.15,
        eta_core_boost: float = 0.10,
        seed_radius: float = 4.0,

        # transition
        front_ring_inner_ratio: float = 0.65,
        front_ring_outer_ratio: float = 1.15,
        front_motion_suppress: float = 0.02,

        # motion
        downstream_x_px: float = 0.42,
        downstream_y_px: float = 0.42,
        curl_px: float = 0.04,
        sway_px: float = 0.015,
        edge_slow_pow: float = 1.0,
        max_disp_px: float = 3.0,

        # guards
        adv_dilate: int = 12,
        source_guard_dilate: int = 5,
        solid_damping: float = 1.0,

        # noise
        noise_seed: int = 123,
        noise_smooth_k: int = 9,
        noise_smooth_iters: int = 2,

        # breach point：强制指定起始点位置
        # （假设原图入口在左上角，这里的数字是基于 N=64 或 128 的像素坐标，请根据实际调整。例如设置在左上角的 [10, 10] 位置）
        breach_xy: tuple[float, float] | None = (10.0, 10.0), 

        # directional flooding
        flow_dir_xy: tuple[float, float] = (1.0, 1.0),   
        dir_bias_strength: float = 30.0,                 # 大幅减小此值，防止距离场数学畸变
        dir_front_gain: float = 0.75,                   
    ):
        super().__init__(N=N)

        self.image_path = image_path or DEFAULT_IMAGE_PATH
        self.water_mask_path = water_mask_path or DEFAULT_WATER_MASK_PATH
        self.region_mask_path = region_mask_path or DEFAULT_REGION_MASK_PATH
        self.solid_mask_path = solid_mask_path
        self.T = int(T)

        for p in [self.image_path, self.water_mask_path, self.region_mask_path]:
            assert os.path.exists(p), f"Missing file: {p}"
        if self.solid_mask_path is not None:
            assert os.path.exists(self.solid_mask_path), f"Missing file: {p}"

        self.level_start = float(level_start)
        self.level_end = float(level_end)
        self.level_ease_pow = float(level_ease_pow)
        self.eta_band = float(eta_band)
        self.eta_gamma = float(eta_gamma)
        self.eta_core_boost = float(eta_core_boost)
        self.seed_radius = float(seed_radius)

        self.front_ring_inner_ratio = float(front_ring_inner_ratio)
        self.front_ring_outer_ratio = float(front_ring_outer_ratio)
        self.front_motion_suppress = float(front_motion_suppress)

        self.downstream_x_px = float(downstream_x_px)
        self.downstream_y_px = float(downstream_y_px)
        self.curl_px = float(curl_px)
        self.sway_px = float(sway_px)
        self.edge_slow_pow = float(edge_slow_pow)
        self.max_disp_px = float(max_disp_px)

        self.adv_dilate = int(adv_dilate)
        self.source_guard_dilate = int(source_guard_dilate)
        self.solid_damping = float(solid_damping)

        self.noise_seed = int(noise_seed)
        self.noise_smooth_k = int(noise_smooth_k)
        self.noise_smooth_iters = int(noise_smooth_iters)

        self.water0 = _load_mask_png(self.water_mask_path, N)
        self.region = _load_mask_png(self.region_mask_path, N)
        self.land = (1.0 - self.water0).clamp(0.0, 1.0)

        self.solid = None
        if self.solid_mask_path is not None:
            self.solid = _load_mask_png(self.solid_mask_path, N)

        if breach_xy is None:
            cx, cy = _compute_mask_centroid_xy(self.water0)
            # 让决口点往左上挪一点，别放在河道中心
            cx = cx - 6.0
            cy = cy - 6.0
        else:
            cx, cy = float(breach_xy[0]), float(breach_xy[1])

        self.breach_cx = cx
        self.breach_cy = cy

        self.dist_map = _radial_dist_map(N, cx, cy, device=torch.device("cpu"))

        # 方向偏置场：右下方向更容易“先被淹”
        xx, yy = _xy_grid(N, device=torch.device("cpu"))
        fx, fy = flow_dir_xy
        v = torch.tensor([fx, fy], dtype=torch.float32)
        v = v / v.norm().clamp_min(1e-6)

        relx = xx - float(cx)
        rely = yy - float(cy)
        proj = relx * v[0] + rely * v[1]   # 沿右下方向的投影

        # proj 越大，代表越靠右下；我们把有效距离减小 => 更容易先扩张到那里
        self.dir_bias = (proj / max(1.0, float(N))) * float(dir_bias_strength)
        self.dir_front_gain = float(dir_front_gain)

        self.flood_seq = self._precompute_flood_hard()
        self.precomputed_grids = self._precompute_grids()

        self.pos_prompt = (
            "aerial view of severe river flooding, floodwater spreading outward into the right-bottom floodplain, "
            "large inundated farmland, wet overflow zone, partially submerged vegetation, realistic flood expansion"
        )

    def get_default_image(self):
        return Image.open(self.image_path).convert("RGB")

    def get_default_framesteps(self):
        return list(range(self.T))

    def _level_at(self, t: int) -> float:
        t = max(0, min(int(t), self.T - 1))
        s = float(t) / float(max(1, self.T - 1))
        if abs(self.level_ease_pow - 1.0) < 1e-6:
            ease = s
        else:
            ease = 1.0 - (1.0 - s) ** self.level_ease_pow
        return self.level_start + (self.level_end - self.level_start) * ease

    def _dist_eff(self, device):
        dist = self.dist_map.to(device)
        dir_bias = self.dir_bias.to(device)
        # 关键：右下方向距离更“短”，所以更容易蔓延过去
        eff = dist - self.seed_radius - 1.6*dir_bias
        return eff.clamp(min=0.0)

    def get_spatial_eta(self, t: int):
        device = self.XY.device
        level = self._level_at(t)
        dist_eff = self._dist_eff(device)

        eta2d = _soft_eta_from_dist(
            dist_eff,
            level=level,
            band=self.eta_band,
            gamma=self.eta_gamma,
            core_boost=self.eta_core_boost,
        )

        eta = eta2d[None, None, ...] * self.region.to(device) * self.land.to(device)
        return eta

    def get_flow(self, t: int):
        t = max(0, min(int(t), self.T - 1))
        return self.precomputed_grids[t]

    def _precompute_flood_hard(self):
        seq = []
        dist_eff = self._dist_eff(torch.device("cpu"))
        region2d = self.region[0, 0]
        land2d = self.land[0, 0]
        water2d = self.water0[0, 0]

        for t in range(self.T):
            level = self._level_at(t)
            hard_level = max(0.0, level - 0.12 * self.eta_band)
            flooded_land = (dist_eff <= float(hard_level)).float() * region2d * land2d
            hard = torch.clamp(water2d + flooded_land, 0.0, 1.0)
            seq.append(hard[None, None, ...].clone())
        return seq

    def _precompute_grids(self):
        torch.manual_seed(self.noise_seed)

        N = self.N
        XY = self.XY.float()
        device = XY.device
        grids = []

        solid_2d = None
        if self.solid is not None:
            solid_2d = self.solid[0, 0].to(device)

        dist_eff = self._dist_eff(device)

        noise_a = torch.randn(1, 1, N, N, device=device)
        noise_b = torch.randn(1, 1, N, N, device=device)
        noise_a = _smooth_noise(noise_a, self.noise_smooth_k, self.noise_smooth_iters)
        noise_b = _smooth_noise(noise_b, self.noise_smooth_k, self.noise_smooth_iters)

        for t in range(self.T):
            s = float(t) / float(max(1, self.T - 1))
            n = (1.0 - s) * noise_a + s * noise_b

            n_pad = F.pad(n, (1, 1, 1, 1), mode="reflect")
            gx = (n_pad[:, :, 1:-1, 2:] - n_pad[:, :, 1:-1, :-2]) * 0.5
            gy = (n_pad[:, :, 2:, 1:-1] - n_pad[:, :, :-2, 1:-1]) * 0.5

            dx = self.downstream_x_px + self.sway_px * n + self.curl_px * gx
            dy = self.downstream_y_px + self.curl_px * gy
            dx = dx[0, 0]
            dy = dy[0, 0]

            eta = self.get_spatial_eta(t)[0, 0]
            level = self._level_at(t)

            front_ring = _make_transition_ring(
                dist_eff,
                level=level,
                ring_inner_ratio=self.front_ring_inner_ratio,
                ring_outer_ratio=self.front_ring_outer_ratio,
                band=self.eta_band,
            )

            core = (eta > 0.7).float() * eta
            frontier = front_ring * (0.45 + 0.55 * eta)
            adv = frontier + 0.22 * core
            adv = adv.clamp(0.0, 1.0).pow(self.edge_slow_pow)

            ring_suppress = (1.0 - self.front_motion_suppress * front_ring).clamp(0.0, 1.0)
            adv = adv * ring_suppress

            flooded_hard = self.flood_seq[t].to(device)
            adv_mask = _dilate(flooded_hard, self.adv_dilate)[0, 0]
            soft_near = _dilate((eta[None, None, ...] > 0.04).float(), 5)[0, 0]
            move_mask = torch.maximum(adv_mask, soft_near)

            # 进一步给右下方向增加前沿推进
            dx = dx + self.dir_front_gain * front_ring
            dy = dy + self.dir_front_gain * front_ring

            dx = dx * adv * move_mask
            dy = dy * adv * move_mask

            # 不再完全消掉整体漂移，只去掉一部分
            dx, dy = _partial_zero_mean_flow(dx, dy, move_mask, keep_ratio=0.75)

            if solid_2d is not None:
                damping = (1.0 - self.solid_damping * solid_2d).clamp(0.0, 1.0)
                dx = dx * damping
                dy = dy * damping

            disp = torch.sqrt(dx * dx + dy * dy + 1e-6)
            scale = (self.max_disp_px / disp).clamp(max=1.0)
            dx = dx * scale
            dy = dy * scale

            flow = torch.stack([dx, dy], dim=-1)
            coords = XY - flow

            # Source guard 放宽，不然前沿借不到纹理
            source_mask = torch.maximum(
                _dilate(flooded_hard, self.source_guard_dilate)[0, 0],
                (eta > 0.10).float(),
            )
            valid = source_mask[..., None]
            coords = coords * valid + XY * (1.0 - valid)

            grid = coords.clone()
            grid[..., 0] = 2 * (grid[..., 0] / (N - 1)) - 1
            grid[..., 1] = 2 * (grid[..., 1] / (N - 1)) - 1
            grids.append(grid)

        return grids