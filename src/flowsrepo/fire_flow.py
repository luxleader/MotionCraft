import os
import numpy as np
import torch
import torch.nn.functional as F
import PIL.Image as Image

from .base_flow import BaseFlow

try:
    from phi.torch.flow import (
        Box,
        CenteredGrid,
        Solve,
        StaggeredGrid,
        advect,
        diffuse,
        extrapolation,
        fluid,
        math,
        resample,
        spatial,
    )
    PHIFLOW_AVAILABLE = True
except Exception:
    PHIFLOW_AVAILABLE = False


DEFAULT_IMAGE_PATH = "/home/yangdonglin/MotionCraft/src/flowsrepo/data/fire/fire_scene.png"
DEFAULT_FIRE_MASK_PATH = "/home/yangdonglin/MotionCraft/src/flowsrepo/data/fire/fire_mask.png"
DEFAULT_FUEL_MASK_PATH = "/home/yangdonglin/MotionCraft/src/flowsrepo/data/fire/fuel_mask.png"


def _load_mask_png(path: str, N: int):
    m = Image.open(path).convert("L").resize((N, N))
    m = torch.from_numpy(np.array(m)).float() / 255.0
    return (m > 0.5).float()[None, None, ...]


def _dilate(x: torch.Tensor, r: int):
    if r <= 0:
        return x
    k = 2 * r + 1
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=r)


def _coords_to_index(coords_xy: torch.Tensor, N: int) -> torch.Tensor:
    idx = coords_xy.round().long()
    idx[..., 0] = idx[..., 0].clamp(0, N - 1)
    idx[..., 1] = idx[..., 1].clamp(0, N - 1)
    return idx


def _torch_mask_to_phi_grid(mask_2d: torch.Tensor, domain):
    """
    mask_2d: (H, W), torch float tensor on CPU/GPU.
    PhiFlow 里这里用 CenteredGrid 表示标量烟密度 / 火源项。
    第一维是 y，第二维是 x，对应当前仓库里 grid_sample 的坐标约定。
    """
    phi_values = math.tensor(mask_2d.detach().cpu().numpy(), spatial("y,x"))
    return CenteredGrid(
        phi_values,
        extrapolation.BOUNDARY,
        domain,
        x=mask_2d.shape[1],
        y=mask_2d.shape[0],
    )


class FireFlow(BaseFlow):
    """
    用 PhiFlow + Navier-Stokes 生成火焰区域的速度场，并把速度场当作 optical flow。

    设计原则：
    1. 背景不动：未着火区域仍然严格 identity warp。
    2. 火焰重绘：保留 fuel 约束的 burn mask，继续作为 Spatial-eta。
    3. 运动来源改为物理模拟：不再手写 curl / sway / drift 的启发式位移，
       而是用 PhiFlow 的 advection + diffusion + pressure projection 得到速度场。
    """

    def __init__(
        self,
        N: int,
        image_path: str = None,
        fire_mask_path: str = None,
        fuel_mask_path: str = None,
        T: int = 48,

        # ---------- 燃烧区域扩张（用于 eta / source-guard，不是光流本身） ----------
        spread_px_per_frame: int = 3,
        spread_accel: float = 1.0,
        fuel_dilate: int = 4,

        # ---------- PhiFlow 物理参数 ----------
        dt: float = 1.0,
        buoyancy_y: float = 0.16,      # 向上浮力强度（y 负方向为向上）
        wind_x: float = 0.05,          # 水平环境风，>0 向右
        viscosity: float = 0.003,      # 速度场黏性
        smoke_diffusion: float = 0.0005,
        inflow_rate: float = 0.30,     # 每帧注入火源强度
        density_decay: float = 0.985,  # 烟密度衰减，避免全图长期残留
        max_speed_px: float = 3.5,     # 把速度场裁成稳定的像素位移

        # ---------- 保护策略 ----------
        advect_dilate: int = 5,
        source_guard_dilate: int = 1,
        fuel_velocity_damping: float = 1.0,
        smoke_threshold: float = 0.03,
    ):
        super().__init__(N=N)

        if not PHIFLOW_AVAILABLE:
            raise ImportError(
                "FireFlow now depends on PhiFlow. Please install it first, e.g. `pip install phiflow`."
            )

        self.image_path = image_path or DEFAULT_IMAGE_PATH
        self.fire_mask_path = fire_mask_path or DEFAULT_FIRE_MASK_PATH
        self.fuel_mask_path = fuel_mask_path or DEFAULT_FUEL_MASK_PATH
        self.T = int(T)

        for p in [self.image_path, self.fire_mask_path, self.fuel_mask_path]:
            assert os.path.exists(p), f"Missing file: {p}"

        self.spread_px_per_frame = int(spread_px_per_frame)
        self.spread_accel = float(spread_accel)
        self.fuel_dilate = int(fuel_dilate)

        self.dt = float(dt)
        self.buoyancy_y = float(buoyancy_y)
        self.wind_x = float(wind_x)
        self.viscosity = float(viscosity)
        self.smoke_diffusion = float(smoke_diffusion)
        self.inflow_rate = float(inflow_rate)
        self.density_decay = float(density_decay)
        self.max_speed_px = float(max_speed_px)

        self.advect_dilate = int(advect_dilate)
        self.source_guard_dilate = int(source_guard_dilate)
        self.fuel_velocity_damping = float(fuel_velocity_damping)
        self.smoke_threshold = float(smoke_threshold)

        fire0 = _load_mask_png(self.fire_mask_path, N)
        fuel = _load_mask_png(self.fuel_mask_path, N)
        if self.fuel_dilate > 0:
            fuel = _dilate(fuel, self.fuel_dilate)

        self.fire0 = fire0
        self.fuel = fuel

        # 这里继续保留 burn_seq：
        # 它负责“哪里允许持续着火 / 重绘”，而真正的位移方向和速度来自 PhiFlow。
        self.burn_seq = self._precompute_burn()
        self.precomputed_eta = self.burn_seq
        self.precomputed_grids = self._precompute_grids()

        self.pos_prompt = (
            "aerial view of buildings on fire, flames and hot smoke spreading between rooftops, "
            "realistic fire, cinematic, detailed"
        )

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
        if self.T <= 1:
            return max(1, self.spread_px_per_frame)
        rate = self.spread_px_per_frame * (1.0 + self.spread_accel * (float(t) / float(self.T - 1)))
        return max(1, int(round(rate)))

    def _precompute_burn(self):
        """
        继续保留 fuel 约束的蔓延逻辑：
        - 这不是物理速度场本身
        - 它用于 Spatial-eta / source guard / 限制背景不被错误卷入
        """
        region = self.fire0.clone()
        burn_seq = []
        for t in range(self.T):
            rate = self._frame_spread_rate(t)
            grow = _dilate(region, rate)
            grow_fuel = grow * self.fuel
            region = torch.clamp(region + grow_fuel, 0.0, 1.0)
            burn_seq.append(region.clone())
        return burn_seq

    def _simulate_phiflow_velocity(self):
        """
        用 PhiFlow 做一个二维 Eulerian smoke plume：
        - smoke: 标量密度场
        - velocity: StaggeredGrid 速度场
        - 每步做 advection -> diffusion -> buoyancy -> incompressibility
        输出每帧中心采样后的速度场 (H, W, 2)，后面直接转成 optical flow 用。
        """
        N = self.N
        domain = Box(x=float(N), y=float(N))

        smoke = _torch_mask_to_phi_grid(self.fire0[0, 0], domain)
        velocity = StaggeredGrid(0, 0, domain, x=N, y=N)
        pressure = None

        trajectories = []

        for t in range(self.T):
            burn_t = self.burn_seq[t][0, 0]
            burn_grid = _torch_mask_to_phi_grid(burn_t, domain)

            # 1) Advect / diffuse smoke density
            smoke = advect.mac_cormack(smoke, velocity, dt=self.dt)
            if self.smoke_diffusion > 0:
                smoke = diffuse.explicit(smoke, self.smoke_diffusion, self.dt)

            # 2) 持续在当前燃烧区注入密度，模拟持续火源
            smoke = smoke * self.density_decay + self.inflow_rate * burn_grid

            # 3) 浮力 + 水平风
            #    y 负方向表示图像坐标中的“向上”
            buoyancy = resample(smoke * (self.wind_x, -self.buoyancy_y), to=velocity)

            # 4) Navier-Stokes operator splitting
            velocity = advect.semi_lagrangian(velocity, velocity, dt=self.dt)
            if self.viscosity > 0:
                velocity = diffuse.explicit(velocity, self.viscosity, self.dt)
            velocity = velocity + buoyancy * self.dt
            velocity, pressure = fluid.make_incompressible(
                velocity,
                (),
                Solve(x0=pressure, rank_deficiency=0),
            )

            # 5) 取中心采样，变成 (H, W, 2) 的 torch tensor
            velocity_center = velocity.at_centers()
            vel_native = velocity_center.values.native("y,x,vector")
            vel_torch = torch.as_tensor(vel_native, dtype=torch.float32).cpu()

            smoke_native = smoke.values.native("y,x")
            smoke_torch = torch.as_tensor(smoke_native, dtype=torch.float32).cpu()

            trajectories.append((smoke_torch, vel_torch))

        return trajectories

    def _precompute_grids(self):
        N = self.N
        XY = self.XY.float()
        fuel_2d = self.fuel[0, 0].float()
        grids = []

        sim_traj = self._simulate_phiflow_velocity()

        for t, (smoke_t, vel_t) in enumerate(sim_traj):
            burn = self.burn_seq[t]
            adv_mask = _dilate(burn, self.advect_dilate)[0, 0].float()

            # ------------------------------
            # 物理模拟的 velocity field -> optical flow (pixel displacement)
            # ------------------------------
            flow = vel_t.clone()
            flow = torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)

            # 只允许火场附近出现位移
            flow = flow * adv_mask[..., None]

            # 在 fuel 区域内把速度压掉，避免整块屋顶/树冠被拖拽变形
            if self.fuel_velocity_damping > 0:
                damping = 1.0 - self.fuel_velocity_damping * fuel_2d
                damping = damping.clamp(0.0, 1.0)
                flow = flow * damping[..., None]

            # 只有烟密度或燃烧 mask 足够强的区域才允许明显运动
            if self.smoke_threshold > 0:
                smoke_gate = (smoke_t > self.smoke_threshold).float()
                smoke_gate = torch.maximum(smoke_gate, adv_mask)
                flow = flow * smoke_gate[..., None]

            flow = flow.clamp(min=-self.max_speed_px, max=self.max_speed_px)

            coords = XY - flow

            # 严格背景保护：在坐标生成阶段直接分界，背景区保持恒等映射
            # adv_mask shape: (N, N)；[..., None] 扩展为 (N, N, 1) 以与 coords/XY (N, N, 2) 广播
            # 与 FloodFlow 保持一致的早期掩码策略
            coords = coords * adv_mask[..., None] + XY * (1.0 - adv_mask[..., None])

            grid = coords.clone()
            grid[..., 0] = 2.0 * (grid[..., 0] / (N - 1.0)) - 1.0
            grid[..., 1] = 2.0 * (grid[..., 1] / (N - 1.0)) - 1.0
            grids.append(grid)

        return grids
