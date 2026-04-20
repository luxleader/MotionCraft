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
DEFAULT_SMOKE_MASK_PATH = "/home/yangdonglin/MotionCraft/src/flowsrepo/data/fire/smoke_mask.png"


def _load_mask_png(path: str, N: int):
    m = Image.open(path).convert("L").resize((N, N))
    m = torch.from_numpy(np.array(m)).float() / 255.0
    return (m > 0.5).float()[None, None, ...]


def _dilate(x: torch.Tensor, r: int):
    if r <= 0:
        return x
    k = 2 * r + 1
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=r)


def _shift_mask(x: torch.Tensor, dy: int = 0, dx: int = 0):
    """
    零填充平移，不使用 roll，避免从边界绕回。
    x: (B, C, H, W)
    dy < 0 表示向上移动，dx < 0 表示向左移动。
    """
    if dy == 0 and dx == 0:
        return x

    out = torch.zeros_like(x)
    H, W = x.shape[-2:]

    dst_y0 = max(dy, 0)
    dst_y1 = H + min(dy, 0)
    src_y0 = max(-dy, 0)
    src_y1 = H - max(dy, 0)

    dst_x0 = max(dx, 0)
    dst_x1 = W + min(dx, 0)
    src_x0 = max(-dx, 0)
    src_x1 = W - max(dx, 0)

    if dst_y0 < dst_y1 and dst_x0 < dst_x1:
        out[..., dst_y0:dst_y1, dst_x0:dst_x1] = x[..., src_y0:src_y1, src_x0:src_x1]
    return out


def _torch_mask_to_phi_grid(mask_2d: torch.Tensor, domain):
    """
    mask_2d: (H, W), torch float tensor on CPU/GPU.
    用 CenteredGrid 表示标量场（火焰温度 / 烟密度 / 源项）。
    第一维是 y，第二维是 x，与当前仓库里的 grid_sample 坐标约定一致。
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
    用 PhiFlow 生成共享速度场，让火焰和烟雾在同一套物理运动下共同演化。

    这版的两个重点：
    1. 火焰蔓延带有明确方向偏置，可控制为“向左上角蔓延”。
    2. 烟雾不是初始就大面积存在，而是在火焰出现之后再逐步生成，
       使视觉上更接近“先有火，再有烟”。
    """

    def __init__(
        self,
        N: int,
        image_path: str = None,
        fire_mask_path: str = None,
        fuel_mask_path: str = None,
        smoke_mask_path: str = None,
        T: int = 24,

        # ---------- 燃烧区域扩张（用于 source guard / eta，不是光流本身） ----------
        spread_px_per_frame: int = 12,
        spread_accel: float = 2.0,
        fuel_dilate: int = 16,
        spread_bias_x: float = -1.0,          # <0 向左，>0 向右
        spread_bias_y: float = -1.0,          # <0 向上，>0 向下
        directional_spread_scale: float = 1.25,
        isotropic_spread_ratio: float = 0.45,

        # ---------- PhiFlow 物理参数 ----------
        dt: float = 1.0,
        buoyancy_y: float = 0.20,             # 最终会转成 -y 方向，即向上浮升
        wind_x: float = -0.18,                # <0 向左，>0 向右
        viscosity: float = 0.003,
        flame_diffusion: float = 0.0002,
        smoke_diffusion: float = 0.0008,
        inflow_rate: float = 0.64,            # 火焰/高温源项注入
        flame_decay: float = 0.965,
        density_decay: float = 0.988,         # 烟密度衰减
        smoke_from_fire: float = 0.16,        # 火焰转化为烟雾的强度
        smoke_source_blend: float = 0.04,     # 在燃烧源附近补一点烟源，避免 plume 断裂
        buoyancy_from_fire: float = 1.00,
        buoyancy_from_smoke: float = 0.55,
        max_speed_px: float = 3.5,
        smoke_start_frame: int = 2,
        smoke_ramp_frames: int = 4,
        initial_smoke_strength: float = 0.0,

        # ---------- 保护与可视化策略 ----------
        advect_dilate: int = 5,
        source_guard_dilate: int = 1,
        eta_dilate: int = 6,
        fuel_velocity_damping: float = 1.0,
        smoke_threshold: float = 0.03,
        flame_threshold: float = 0.04,
        eta_smoke_threshold: float = 0.015,
    ):
        super().__init__(N=N)

        if not PHIFLOW_AVAILABLE:
            raise ImportError(
                "FireFlow now depends on PhiFlow. Please install it first, e.g. `pip install phiflow`."
            )

        self.image_path = image_path or DEFAULT_IMAGE_PATH
        self.fire_mask_path = fire_mask_path or DEFAULT_FIRE_MASK_PATH
        self.fuel_mask_path = fuel_mask_path or DEFAULT_FUEL_MASK_PATH
        self.smoke_mask_path = smoke_mask_path or DEFAULT_SMOKE_MASK_PATH
        self.T = int(T)

        for p in [
            self.image_path,
            self.fire_mask_path,
            self.fuel_mask_path,
            self.smoke_mask_path,
        ]:
            assert os.path.exists(p), f"Missing file: {p}"

        self.spread_px_per_frame = int(spread_px_per_frame)
        self.spread_accel = float(spread_accel)
        self.fuel_dilate = int(fuel_dilate)
        self.spread_bias_x = float(spread_bias_x)
        self.spread_bias_y = float(spread_bias_y)
        self.directional_spread_scale = float(directional_spread_scale)
        self.isotropic_spread_ratio = float(isotropic_spread_ratio)

        self.dt = float(dt)
        self.buoyancy_y = float(buoyancy_y)
        self.wind_x = float(wind_x)
        self.viscosity = float(viscosity)
        self.flame_diffusion = float(flame_diffusion)
        self.smoke_diffusion = float(smoke_diffusion)
        self.inflow_rate = float(inflow_rate)
        self.flame_decay = float(flame_decay)
        self.density_decay = float(density_decay)
        self.smoke_from_fire = float(smoke_from_fire)
        self.smoke_source_blend = float(smoke_source_blend)
        self.buoyancy_from_fire = float(buoyancy_from_fire)
        self.buoyancy_from_smoke = float(buoyancy_from_smoke)
        self.max_speed_px = float(max_speed_px)
        self.smoke_start_frame = int(smoke_start_frame)
        self.smoke_ramp_frames = int(smoke_ramp_frames)
        self.initial_smoke_strength = float(initial_smoke_strength)

        self.advect_dilate = int(advect_dilate)
        self.source_guard_dilate = int(source_guard_dilate)
        self.eta_dilate = int(eta_dilate)
        self.fuel_velocity_damping = float(fuel_velocity_damping)
        self.smoke_threshold = float(smoke_threshold)
        self.flame_threshold = float(flame_threshold)
        self.eta_smoke_threshold = float(eta_smoke_threshold)

        fire0 = _load_mask_png(self.fire_mask_path, N)
        fuel = _load_mask_png(self.fuel_mask_path, N)
        smoke_mask = _load_mask_png(self.smoke_mask_path, N)
        if self.fuel_dilate > 0:
            fuel = _dilate(fuel, self.fuel_dilate)

        self.fire0 = fire0
        self.fuel = fuel
        self.smoke_mask = smoke_mask

        # 关键改动：不再把整张 smoke_mask 当作 t=0 的烟密度。
        # 这样序列开头会先出现火焰，烟雾再由火焰逐步生成。
        self.smoke0 = (self.fire0 * self.initial_smoke_strength).clamp(0.0, 1.0)

        # burn_seq 继续负责“哪里允许持续着火 / 重绘”，
        # 但真正的运动方向和速度由耦合物理场给出。
        self.burn_seq = self._precompute_burn()

        sim_traj = self._simulate_coupled_plume()
        self.precomputed_eta = self._precompute_eta(sim_traj)
        self.precomputed_grids = self._precompute_grids(sim_traj)

        self.pos_prompt = (
            "aerial view of wildfire spreading to the upper-left, flames appear first and smoke follows, "
            "realistic fire plume, cinematic, detailed"
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

    def _directional_shift(self, rate: int):
        vec = np.array([self.spread_bias_x, self.spread_bias_y], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return 0, 0
        vec = vec / norm
        step = max(1, int(round(rate * self.directional_spread_scale)))
        dx = int(round(step * float(vec[0])))
        dy = int(round(step * float(vec[1])))
        return dy, dx

    def _smoke_release_scale(self, t: int) -> float:
        if t < self.smoke_start_frame:
            return 0.0
        if self.smoke_ramp_frames <= 0:
            return 1.0
        return min(1.0, float(t - self.smoke_start_frame + 1) / float(self.smoke_ramp_frames))

    def _precompute_burn(self):
        """
        fuel 约束的蔓延逻辑：
        - 这不是物理速度场本身
        - 它用于 Spatial-eta / source guard / 限制背景不被错误卷入
        - 与旧版不同：这里加入方向偏置，让燃烧前沿优先向左上角扩张
        """
        region = self.fire0.clone()
        burn_seq = []
        for t in range(self.T):
            rate = self._frame_spread_rate(t)
            iso_rate = max(1, int(round(rate * self.isotropic_spread_ratio)))
            grow_iso = _dilate(region, iso_rate)

            dy, dx = self._directional_shift(rate)
            shifted = _shift_mask(region, dy=dy, dx=dx)
            grow_dir = _dilate(shifted, max(1, rate))

            grow = torch.maximum(grow_iso, grow_dir)
            grow_fuel = grow * self.fuel
            region = torch.clamp(region + grow_fuel, 0.0, 1.0)
            burn_seq.append(region.clone())
        return burn_seq

    def _simulate_coupled_plume(self):
        """
        用共享 velocity 同时演化 fire 和 smoke 两个标量场。

        fire field：
            更接近高温/火焰核，扩散更弱、衰减更快，持续由 burn mask 注入。
        smoke field：
            不再一开始就存在，而是在火焰建立之后，按时间逐步生成。

        二者通过同一个 velocity 相互耦合：
            - fire / smoke 都被同一 velocity advect
            - buoyancy 由 fire + smoke 的组合共同决定
        """
        N = self.N
        domain = Box(x=float(N), y=float(N))

        fire = _torch_mask_to_phi_grid(self.fire0[0, 0], domain)
        smoke = _torch_mask_to_phi_grid(self.smoke0[0, 0], domain)
        velocity = StaggeredGrid(0, 0, domain, x=N, y=N)
        pressure = None

        trajectories = []

        for t in range(self.T):
            burn_t = self.burn_seq[t][0, 0]
            burn_grid = _torch_mask_to_phi_grid(burn_t, domain)

            # 1) 共享速度场下，先 advect 两个标量场
            fire = advect.mac_cormack(fire, velocity, dt=self.dt)
            smoke = advect.mac_cormack(smoke, velocity, dt=self.dt)

            # 2) 各自扩散与衰减
            if self.flame_diffusion > 0:
                fire = diffuse.explicit(fire, self.flame_diffusion, self.dt)
            if self.smoke_diffusion > 0:
                smoke = diffuse.explicit(smoke, self.smoke_diffusion, self.dt)

            fire = fire * self.flame_decay + self.inflow_rate * burn_grid

            # 关键改动：烟雾滞后于火焰生成，而不是 t=0 就整片铺开。
            smoke_release = self._smoke_release_scale(t)
            smoke_source = smoke_release * (
                self.smoke_from_fire * fire + self.smoke_source_blend * burn_grid
            )
            smoke = smoke * self.density_decay + smoke_source

            # 3) buoyancy 由 fire + smoke 共同决定，因此二者运动具有一致物理来源
            plume_drive = (
                self.buoyancy_from_fire * fire
                + self.buoyancy_from_smoke * smoke
            )
            buoyancy = resample(plume_drive * (self.wind_x, -self.buoyancy_y), to=velocity)

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

            # 5) 导出中心采样后的 torch tensor
            velocity_center = velocity.at_centers()
            vel_native = velocity_center.values.native("y,x,vector")
            vel_torch = torch.as_tensor(vel_native, dtype=torch.float32).cpu()

            fire_native = fire.values.native("y,x")
            fire_torch = torch.as_tensor(fire_native, dtype=torch.float32).cpu()

            smoke_native = smoke.values.native("y,x")
            smoke_torch = torch.as_tensor(smoke_native, dtype=torch.float32).cpu()

            trajectories.append((fire_torch, smoke_torch, vel_torch))

        return trajectories

    def _precompute_eta(self, sim_traj):
        """
        Spatial-eta 同时覆盖：
        - 持续燃烧区（burn_seq）
        - plume support（火焰/烟雾上升区域）

        这样可以避免：
        流场已经把烟推上去了，但 diffusion model 仍然只在地表 burn 区域重绘。
        """
        eta_seq = []

        for t, (fire_t, smoke_t, _vel_t) in enumerate(sim_traj):
            burn_t = _dilate(self.burn_seq[t], self.source_guard_dilate)

            fire_soft = fire_t / (fire_t.max() + 1e-6)
            smoke_soft = smoke_t / (smoke_t.max() + 1e-6)
            plume_soft = torch.maximum(fire_soft, smoke_soft)
            plume_soft = plume_soft[None, None].clamp(0.0, 1.0)

            plume_hard = (plume_soft > self.eta_smoke_threshold).float()
            plume_hard = _dilate(plume_hard, self.eta_dilate)

            eta_t = torch.maximum(burn_t, torch.maximum(plume_soft, plume_hard * 0.75))
            eta_t = eta_t.clamp(0.0, 1.0)
            eta_seq.append(eta_t)

        return eta_seq

    def _precompute_grids(self, sim_traj):
        N = self.N
        XY = self.XY.float()
        fuel_2d = self.fuel[0, 0].float()
        grids = []

        for t, (fire_t, smoke_t, vel_t) in enumerate(sim_traj):
            burn_t = self.burn_seq[t]
            eta_t = self.precomputed_eta[t]

            motion_support = torch.maximum(_dilate(burn_t, self.advect_dilate), eta_t)
            adv_mask = (motion_support > 0.05).float()[0, 0]

            flow = vel_t.clone()
            flow = torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
            flow = flow * adv_mask[..., None]

            if self.fuel_velocity_damping > 0:
                damping = 1.0 - self.fuel_velocity_damping * fuel_2d
                damping = damping.clamp(0.0, 1.0)
                flow = flow * damping[..., None]

            fire_gate = (fire_t > self.flame_threshold).float()
            smoke_gate = (smoke_t > self.smoke_threshold).float()
            motion_gate = torch.maximum(torch.maximum(fire_gate, smoke_gate), adv_mask)
            flow = flow * motion_gate[..., None]

            flow = flow.clamp(min=-self.max_speed_px, max=self.max_speed_px)

            coords = XY - flow
            coords = coords * adv_mask[..., None] + XY * (1.0 - adv_mask[..., None])

            grid = coords.clone()
            grid[..., 0] = 2.0 * (grid[..., 0] / (N - 1.0)) - 1.0
            grid[..., 1] = 2.0 * (grid[..., 1] / (N - 1.0)) - 1.0
            grids.append(grid)

        return grids
