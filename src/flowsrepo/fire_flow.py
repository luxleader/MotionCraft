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

        # ---------- 火焰-烟雾耦合参数 ----------
        fire_advect_factor: float = 0.5,
        # 速度场驱动的火焰区域平流权重 (0=纯静态burn_seq, 1=纯速度场平流)
        velocity_flame_coupling: float = 0.3,
        # 局部流速对燃烧速率的影响强度 (0=禁用, 1=全耦合)
        vorticity_strength: float = 0.0,
        # 涡量约束强度，增强火焰卷曲运动 (0=禁用)
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

        self.fire_advect_factor = float(fire_advect_factor)
        self.velocity_flame_coupling = float(velocity_flame_coupling)
        self.vorticity_strength = float(vorticity_strength)

        fire0 = _load_mask_png(self.fire_mask_path, N)
        fuel = _load_mask_png(self.fuel_mask_path, N)
        if self.fuel_dilate > 0:
            fuel = _dilate(fuel, self.fuel_dilate)

        self.fire0 = fire0
        self.fuel = fuel

        # 这里继续保留 burn_seq：
        # 它负责“哪里允许持续着火 / 重绘”，而真正的位移方向和速度来自 PhiFlow。
        self.burn_seq = self._precompute_burn()
        # _precompute_grids 同时返回光流网格和 spatial-eta 序列，
        # 当 fire_advect_factor > 0 时 eta 反映速度场驱动的动态火焰区域。
        self.precomputed_grids, self.precomputed_eta = self._precompute_grids()

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

    def _advect_fire_region(self, fire_grid, velocity, domain, N):
        """
        使用与烟雾相同的速度场对火焰区域进行半拉格朗日平流。
        确保火焰和烟雾受到同一物理力的约束，保持运动同步。

        Args:
            fire_grid: PhiFlow CenteredGrid，当前火焰区域（标量密度）
            velocity: PhiFlow StaggeredGrid，当前速度场
            domain: PhiFlow Box，计算域
            N: 网格分辨率

        Returns:
            更新后的 PhiFlow CenteredGrid（火焰区域）
        """
        fire_advected = advect.semi_lagrangian(fire_grid, velocity, dt=self.dt)
        # 平流后截断到 [0,1] 并重建网格，防止数值漂移累积
        fire_np = fire_advected.values.native("y,x")
        fire_clamped = torch.as_tensor(fire_np, dtype=torch.float32).cpu().clamp(0.0, 1.0)
        return _torch_mask_to_phi_grid(fire_clamped, domain)

    def _compute_adaptive_burn_rate(self, vel_mag: torch.Tensor) -> torch.Tensor:
        """
        根据局部速度大小计算自适应燃烧速率。
        速度越大的区域燃烧越剧烈，模拟风驱动火焰的物理效果。

        Args:
            vel_mag: (H, W) 速度幅值张量

        Returns:
            (H, W) 自适应燃烧速率张量
        """
        # 将速度幅值归一化到 [0,1]，以 max_speed_px 为参考
        normalized_speed = (vel_mag / (self.max_speed_px + 1e-6)).clamp(0.0, 1.0)
        # 燃烧速率随局部速度线性增强
        return self.inflow_rate * (1.0 + self.velocity_flame_coupling * normalized_speed)

    def _apply_vorticity_confinement(self, velocity, domain, N):
        """
        计算涡量约束力并叠加到速度场，使火焰产生更真实的卷曲运动。

        基于 Fedkiw et al. (2001) 的涡量约束方法：
          ω  = ∂vy/∂x − ∂vx/∂y       （2D 标量涡量）
          η  = ∇|ω|                    （涡量梯度）
          N̂  = η / |η|                （归一化方向）
          f  = ε × (N̂ × ω)           （约束力）

        Args:
            velocity: PhiFlow StaggeredGrid
            domain: PhiFlow Box
            N: 网格分辨率

        Returns:
            施加涡量约束后的 StaggeredGrid
        """
        vel_center = velocity.at_centers()
        vel_native = vel_center.values.native("y,x,vector")
        vel_t = torch.as_tensor(vel_native, dtype=torch.float32).cpu()

        vx = vel_t[..., 0]  # (H, W)
        vy = vel_t[..., 1]  # (H, W)

        # 2D 涡量：ω = ∂vy/∂x − ∂vx/∂y（中心差分）
        dvydx = torch.zeros_like(vy)
        dvxdy = torch.zeros_like(vx)
        dvydx[:, 1:-1] = (vy[:, 2:] - vy[:, :-2]) * 0.5
        dvxdy[1:-1, :] = (vx[2:, :] - vx[:-2, :]) * 0.5
        omega = dvydx - dvxdy

        # 涡量幅值的梯度 η = ∇|ω|
        omega_abs = omega.abs()
        eta_x = torch.zeros_like(omega_abs)
        eta_y = torch.zeros_like(omega_abs)
        eta_x[:, 1:-1] = (omega_abs[:, 2:] - omega_abs[:, :-2]) * 0.5
        eta_y[1:-1, :] = (omega_abs[2:, :] - omega_abs[:-2, :]) * 0.5

        # 归一化方向 N̂
        eta_mag = (eta_x ** 2 + eta_y ** 2).sqrt() + 1e-8
        nx = eta_x / eta_mag
        ny = eta_y / eta_mag

        # 约束力：f_x = ε·ny·ω，f_y = −ε·nx·ω
        fx = (self.vorticity_strength * ny * omega).numpy()
        fy = (self.vorticity_strength * (-nx) * omega).numpy()

        # 将约束力转为 PhiFlow CenteredGrid 并重采样到 StaggeredGrid
        fx_grid = CenteredGrid(
            math.tensor(fx, spatial("y,x")),
            extrapolation.BOUNDARY, domain, x=N, y=N,
        )
        fy_grid = CenteredGrid(
            math.tensor(fy, spatial("y,x")),
            extrapolation.BOUNDARY, domain, x=N, y=N,
        )
        # 利用 PhiFlow 向量广播：scalar_grid * (cx, cy) → vector CenteredGrid
        confinement = resample(
            fx_grid * (1.0, 0.0) + fy_grid * (0.0, 1.0),
            to=velocity,
        )
        return velocity + confinement * self.dt

    def _simulate_phiflow_velocity(self):
        """
        用 PhiFlow 做一个二维 Eulerian smoke plume，同时跟踪耦合的火焰区域状态：
        - smoke: 标量烟密度场
        - fire_region: 标量火焰密度场（与烟共享同一速度场平流）
        - velocity: StaggeredGrid 速度场
        - 每步做 advection -> diffusion -> buoyancy -> incompressibility

        当 fire_advect_factor > 0 时，火焰区域通过半拉格朗日平流随速度场移动，
        与烟密度保持物理同步；当 velocity_flame_coupling > 0 时，燃烧速率由局部
        速度幅值自适应调节；当 vorticity_strength > 0 时施加涡量约束增强火焰卷曲。

        返回每帧的 (smoke_torch, vel_torch, fire_state_torch) 三元组列表。
        """
        N = self.N
        domain = Box(x=float(N), y=float(N))

        smoke = _torch_mask_to_phi_grid(self.fire0[0, 0], domain)
        velocity = StaggeredGrid(0, 0, domain, x=N, y=N)
        pressure = None

        # 初始化耦合火焰区域（仅在 fire_advect_factor > 0 时使用）
        fire_region = (_torch_mask_to_phi_grid(self.fire0[0, 0], domain)
            if self.fire_advect_factor > 0 else None)

        trajectories = []

        for t in range(self.T):
            burn_t = self.burn_seq[t][0, 0]  # (N, N) 预计算燃烧区域

            # --- 获取当前步初始速度幅值（用于自适应燃烧速率）---
            if self.velocity_flame_coupling > 0:
                vel_c = velocity.at_centers()
                vel_np = vel_c.values.native("y,x,vector")
                vel_cur = torch.as_tensor(vel_np, dtype=torch.float32).cpu()
                vel_mag = vel_cur.norm(dim=-1)  # (N, N)
                adaptive_rate_2d = self._compute_adaptive_burn_rate(vel_mag)

            # --- 火焰区域半拉格朗日平流（与烟同步移动）---
            if self.fire_advect_factor > 0:
                fire_region = self._advect_fire_region(fire_region, velocity, domain, N)

                # 从 PhiFlow 网格提取平流后的火焰区域
                fire_native = fire_region.values.native("y,x")
                fire_torch = torch.as_tensor(fire_native, dtype=torch.float32).cpu()

                # 混合：fire_advect_factor 控制速度平流 vs 预计算扩张的权重
                blended_fire = (
                    self.fire_advect_factor * fire_torch
                    + (1.0 - self.fire_advect_factor) * burn_t
                )
                # 应用燃料约束并截断
                blended_fire = (blended_fire * self.fuel[0, 0]).clamp(0.0, 1.0)

                # 将混合结果写回 fire_region，参与下一帧平流
                fire_region = _torch_mask_to_phi_grid(blended_fire, domain)
                effective_burn_grid = fire_region
                fire_state_t = blended_fire  # (N, N) tensor
            else:
                burn_grid = _torch_mask_to_phi_grid(burn_t, domain)
                effective_burn_grid = burn_grid
                fire_state_t = burn_t  # (N, N) tensor

            # 1) 烟密度平流 / 扩散
            smoke = advect.mac_cormack(smoke, velocity, dt=self.dt)
            if self.smoke_diffusion > 0:
                smoke = diffuse.explicit(smoke, self.smoke_diffusion, self.dt)

            # 2) 在当前燃烧区注入密度（自适应或固定速率）
            if self.velocity_flame_coupling > 0:
                adaptive_rate_grid = _torch_mask_to_phi_grid(adaptive_rate_2d, domain)
                smoke = smoke * self.density_decay + adaptive_rate_grid * effective_burn_grid
            else:
                smoke = smoke * self.density_decay + self.inflow_rate * effective_burn_grid

            # 3) 浮力 + 水平风
            #    y 负方向表示图像坐标中的“向上”
            buoyancy = resample(smoke * (self.wind_x, -self.buoyancy_y), to=velocity)

            # 4) Navier-Stokes operator splitting
            velocity = advect.semi_lagrangian(velocity, velocity, dt=self.dt)
            if self.viscosity > 0:
                velocity = diffuse.explicit(velocity, self.viscosity, self.dt)
            velocity = velocity + buoyancy * self.dt

            # 5) 可选涡量约束（增强火焰卷曲）
            if self.vorticity_strength > 0:
                velocity = self._apply_vorticity_confinement(velocity, domain, N)

            velocity, pressure = fluid.make_incompressible(
                velocity,
                (),
                Solve(x0=pressure, rank_deficiency=0),
            )

            # 6) 取中心采样，变成 (H, W, 2) 的 torch tensor
            velocity_center = velocity.at_centers()
            vel_native = velocity_center.values.native("y,x,vector")
            vel_torch = torch.as_tensor(vel_native, dtype=torch.float32).cpu()

            smoke_native = smoke.values.native("y,x")
            smoke_torch = torch.as_tensor(smoke_native, dtype=torch.float32).cpu()

            trajectories.append((smoke_torch, vel_torch, fire_state_t))

        return trajectories

    def _precompute_grids(self):
        """
        计算所有帧的采样网格（optical flow 映射表）并同时生成 spatial-eta 序列。

        当 fire_advect_factor > 0 时，advection mask 及 eta 使用物理模拟驱动的
        动态火焰区域，使光流掩码与底层速度场保持同步；否则回退到静态 burn_seq。

        Returns:
            grids: List[Tensor (N,N,2)]，每帧的归一化采样坐标
            fire_states: List[Tensor (1,1,N,N)]，每帧的 spatial-eta 遮罩
        """
        N = self.N
        XY = self.XY.float()
        fuel_2d = self.fuel[0, 0].float()
        grids = []
        fire_states = []

        sim_traj = self._simulate_phiflow_velocity()

        for t, (smoke_t, vel_t, fire_state_t) in enumerate(sim_traj):
            # 选择 advection mask 来源：
            #   fire_advect_factor > 0 → 使用速度场驱动的动态火焰区域
            #   fire_advect_factor == 0 → 使用静态预计算 burn_seq（原始行为）
            if self.fire_advect_factor > 0:
                fire_4d = fire_state_t[None, None, ...]  # (1,1,N,N)
                adv_mask = _dilate(fire_4d, self.advect_dilate)[0, 0].float()
                fire_states.append(fire_4d)
            else:
                burn = self.burn_seq[t]
                adv_mask = _dilate(burn, self.advect_dilate)[0, 0].float()
                fire_states.append(burn)

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

        return grids, fire_states
