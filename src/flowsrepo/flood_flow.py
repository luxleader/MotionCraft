import os
import numpy as np
import torch
import torch.nn.functional as F
import PIL.Image as Image
from .base_flow import BaseFlow

# ================== 路径（按你机器改） ==================
DEFAULT_IMAGE_PATH = "/home/yangdonglin/MotionCraft/src/flowsrepo/data/water/water_scene.png"
DEFAULT_WATER_MASK_PATH = "/home/yangdonglin/MotionCraft/src/flowsrepo/data/water/water_mask.png"
DEFAULT_REGION_MASK_PATH = "/home/yangdonglin/MotionCraft/src/flowsrepo/data/water/region_mask.png"
DEFAULT_SOLID_MASK_PATH = "/home/yangdonglin/MotionCraft/src/flowsrepo/data/water/solid_mask.png"  # 可选，没有就传 None


# ================== 工具函数 ==================
def _load_mask_png(path: str, N: int):
    m = Image.open(path).convert("L").resize((N, N))
    m = torch.from_numpy(np.array(m)).float() / 255.0
    return (m > 0.5).float()[None, None, ...]  # (1,1,N,N)


def _dilate(x: torch.Tensor, r: int):
    if r <= 0:
        return x
    k = 2 * r + 1
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=r)


def _erode(x: torch.Tensor, r: int):
    if r <= 0:
        return x
    k = 2 * r + 1
    return -F.max_pool2d(-x, kernel_size=k, stride=1, padding=r)


def _smooth_noise(x, k=9, iters=1):
    for _ in range(iters):
        x = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
    return x


def _compute_mask_centroid_xy(mask_01: torch.Tensor) -> tuple[float, float]:
    m = (mask_01[0, 0] > 0.5).float()
    ys, xs = torch.where(m > 0.5)
    if xs.numel() == 0:
        N = mask_01.shape[-1]
        return (0.5 * (N - 1), 0.5 * (N - 1))
    cx = xs.float().mean().item()
    cy = ys.float().mean().item()
    return cx, cy


def _radial_dist_map(N: int, cx: float, cy: float, device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(N, device=device, dtype=torch.float32),
        torch.arange(N, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.sqrt((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2)


def _smoothstep01(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _soft_eta_from_dist(
    dist_2d: torch.Tensor,
    level: float,
    band: float,
    gamma: float = 1.0,
    core_boost: float = 0.0,
) -> torch.Tensor:
    band = max(1e-6, float(band))
    x = (float(level) - dist_2d) / band + 0.5
    x = _smoothstep01(x)

    if gamma != 1.0:
        x = x.pow(float(gamma))

    if core_boost > 0.0:
        x = (x + core_boost * x * x).clamp(0.0, 1.0)

    return x


def _make_transition_ring(
    dist_2d: torch.Tensor,
    level: float,
    ring_inner_ratio: float,
    ring_outer_ratio: float,
    band: float,
) -> torch.Tensor:
    inner = level - ring_inner_ratio * band
    outer = level + ring_outer_ratio * band

    left = (dist_2d - inner) / max(1e-6, (level - inner))
    left = _smoothstep01(left)

    right = 1.0 - (dist_2d - level) / max(1e-6, (outer - level))
    right = _smoothstep01(right)

    ring = torch.minimum(left, right).clamp(0.0, 1.0)
    return ring


def _sample_mask_with_grid(mask_2d: torch.Tensor, grid_nxn2: torch.Tensor) -> torch.Tensor:
    """
    用连续采样代替 round 索引，减少“色块偏移/块状跳变”。
    mask_2d: (N,N)
    grid_nxn2: (N,N,2), 已经是 [-1,1]
    return: (N,N)
    """
    src = mask_2d[None, None, ...].float()
    grid = grid_nxn2[None, ...].float()
    val = F.grid_sample(
        src,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return val[0, 0]


def _zero_mean_flow(dx: torch.Tensor, dy: torch.Tensor, weight: torch.Tensor):
    """
    去掉有效区域内的整体平移分量，避免整块纹理一直往同一方向滑。
    """
    w = weight.clamp(0.0, 1.0)
    denom = w.sum().clamp_min(1e-6)
    mx = (dx * w).sum() / denom
    my = (dy * w).sum() / denom
    return dx - mx, dy - my


# ================== FloodFlow ==================
class FloodFlow(BaseFlow):
    def __init__(
        self,
        N: int,
        image_path: str = None,
        water_mask_path: str = None,
        region_mask_path: str = None,
        solid_mask_path: str = None,
        T: int = 48,

        # ---------- flooding (radial) ----------
        level_start: float = 8.0,
        level_end: float = 220.0,
        level_ease_pow: float = 1.8,
        eta_band: float = 40.0,
        eta_gamma: float = 1.5,
        eta_core_boost: float = 0.08,
        seed_radius: float = 8.0,

        # ---------- transition/front ring ----------
        front_ring_inner_ratio: float = 0.8,
        front_ring_outer_ratio: float = 1.5,
        front_motion_suppress: float = 0.35,

        # ---------- motion ----------
        # 关键：把全局平移降到很小，避免整块色块滑走
        downstream_x_px: float = 0.03,
        downstream_y_px: float = 0.03,
        curl_px: float = 0.10,
        sway_px: float = 0.04,
        edge_slow_pow: float = 1.25,
        max_disp_px: float = 0.65,   # 限制单帧最大位移，防止滑移太明显

        # ---------- guards ----------
        adv_dilate: int = 8,
        source_guard_dilate: int = 4,
        solid_damping: float = 1.0,

        # ---------- noise ----------
        noise_seed: int = 123,
        noise_smooth_k: int = 9,
        noise_smooth_iters: int = 2,

        # ---------- breach point ----------
        breach_xy: tuple[float, float] | None = None,
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
            assert os.path.exists(self.solid_mask_path), f"Missing file: {self.solid_mask_path}"

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
        else:
            cx, cy = float(breach_xy[0]), float(breach_xy[1])

        self.breach_cx = cx
        self.breach_cy = cy
        self.dist_map = _radial_dist_map(N, cx, cy, device=torch.device("cpu"))

        self.flood_seq = self._precompute_flood_hard()
        self.precomputed_grids = self._precompute_grids()

        self.pos_prompt = (
            "aerial view of river flooding, shallow floodwater spreading from the river bank, "
            "wet transitional zone, partially submerged grassland, realistic floodplain"
        )

    def get_default_image(self):
        return Image.open(self.image_path).convert("RGB")

    def get_default_framesteps(self):
        return list(range(self.T))

    def _level_at(self, t: int) -> float:
        t = max(0, min(int(t), self.T - 1))
        s = float(t) / float(max(1, self.T - 1))
        ease = 1.0 - (1.0 - s) ** self.level_ease_pow
        return self.level_start + (self.level_end - self.level_start) * ease

    def _dist_eff(self, device):
        dist = self.dist_map.to(device)
        return (dist - self.seed_radius).clamp(min=0.0)

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
        dist = self.dist_map
        region2d = self.region[0, 0]
        land2d = self.land[0, 0]
        water2d = self.water0[0, 0]

        for t in range(self.T):
            level = self._level_at(t)
            dist_eff = (dist - self.seed_radius).clamp(min=0.0)

            hard_level = max(0.0, level - 0.18 * self.eta_band)
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

        # 两张不同噪声，做轻微时序插值，避免每帧都像同一块纹理在滑
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

            # 关键：主要让“前沿”动，已淹核心区更稳
            core = (eta > 0.7).float() * eta
            frontier = front_ring * (0.35 + 0.65 * eta)
            adv = frontier + 0.12 * core
            adv = adv.clamp(0.0, 1.0).pow(self.edge_slow_pow)

            ring_suppress = (1.0 - self.front_motion_suppress * front_ring).clamp(0.0, 1.0)
            adv = adv * ring_suppress

            # 有效运动区域
            flooded_hard = self.flood_seq[t].to(device)
            adv_mask = _dilate(flooded_hard, self.adv_dilate)[0, 0]
            soft_near = _dilate((eta[None, None, ...] > 0.05).float(), 3)[0, 0]
            move_mask = torch.maximum(adv_mask, soft_near)

            dx = dx * adv * move_mask
            dy = dy * adv * move_mask

            # 去掉整体平移分量，防止“整块色块偏移”
            dx, dy = _zero_mean_flow(dx, dy, move_mask)

            # 固体锁死
            if solid_2d is not None:
                damping = (1.0 - self.solid_damping * solid_2d).clamp(0.0, 1.0)
                dx = dx * damping
                dy = dy * damping

            # 限制单帧最大位移
            disp = torch.sqrt(dx * dx + dy * dy + 1e-6)
            scale = (self.max_disp_px / disp).clamp(max=1.0)
            dx = dx * scale
            dy = dy * scale

            flow = torch.stack([dx, dy], dim=-1)
            coords = XY - flow

            grid = coords.clone()
            grid[..., 0] = 2 * (grid[..., 0] / (N - 1)) - 1
            grid[..., 1] = 2 * (grid[..., 1] / (N - 1)) - 1

            # 用连续采样替代 round 索引，减少块状偏移
            source_mask = _dilate(flooded_hard, self.source_guard_dilate)[0, 0]
            source_soft = _erode((eta[None, None, ...] > 0.18).float(), 1)[0, 0]
            source_mask = torch.maximum(source_mask, source_soft)

            valid = _sample_mask_with_grid(source_mask, grid)
            valid = (valid > 0.35).float()[..., None]

            coords = coords * valid + XY * (1.0 - valid)

            grid = coords.clone()
            grid[..., 0] = 2 * (grid[..., 0] / (N - 1)) - 1
            grid[..., 1] = 2 * (grid[..., 1] / (N - 1)) - 1
            grids.append(grid)

        return grids