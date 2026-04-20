import torch
import torch.nn.functional as F
import PIL.Image as Image
import numpy as np
from .base_flow import BaseFlow


class CanalFloodFlow(BaseFlow):
    """
    面向当前这张“农田 + 斜向沟渠”的俯拍图。

    目标：
    1. 先让河道/沟渠本身出现明显变化（浑浊、涨水、表面流动）
    2. 再从两岸逐步向外漫溢
    3. 尽量少影响道路与右侧深树林
    """

    def __init__(
        self,
        N: int,
        num_frames: int = 24,
        max_spread_px_512: float = 108.0,
        canal_half_width_px_512: float = 22.0,
        canal_core_extra_px_512: float = 7.0,
        road_slowdown: float = 0.34,
        warp_step_px_512: float = 1.15,
        channel_flow_px_512: float = 1.45,
    ) -> None:
        super().__init__(N=N)

        self.num_frames = num_frames
        self.image_path = f"{self.this_path}/data/canal_flood/farm_scene.png"
        self.pos_prompt = (
            "a realistic aerial drone photo of a swollen muddy canal after heavy rain, "
            "turbid flood water filling the canal, rising water level, wet riverbanks, "
            "flood water gradually overflowing into nearby crop rows, partially submerged plants, "
            "natural agricultural flood scene, realistic texture"
        )

        base_img = Image.open(self.image_path).convert("RGB")
        base_img = base_img.resize((N, N), Image.BILINEAR)
        self.base_img = base_img
        base_np = np.array(base_img).astype(np.float32) / 255.0
        base_t = torch.from_numpy(base_np)
        R = base_t[..., 0]
        G = base_t[..., 1]
        B = base_t[..., 2]

        s = float(N) / 512.0

        # 这两个点对应图中的斜向河道中心线
        p0 = torch.tensor([28.0, 404.0], dtype=torch.float32) * s
        p1 = torch.tensor([404.0, 28.0], dtype=torch.float32) * s

        P = self.XY.to(torch.float32)
        d = p1 - p0
        d_norm = torch.clamp(torch.linalg.norm(d), min=1e-6)
        d_unit = d / d_norm

        rel = P - p0
        t_line = (rel * d_unit).sum(dim=-1).clamp(0.0, d_norm)
        proj = p0 + t_line.unsqueeze(-1) * d_unit
        perp = P - proj
        dist_to_center = torch.linalg.norm(perp, dim=-1)

        # 道路位于河道左上侧，作为减速/保护侧
        road_normal = torch.tensor([d_unit[1], -d_unit[0]], dtype=torch.float32)
        signed_side = (perp * road_normal).sum(dim=-1)

        canal_half_width = canal_half_width_px_512 * s
        canal_core_half_width = (canal_half_width_px_512 + canal_core_extra_px_512) * s
        max_spread = max_spread_px_512 * s
        channel_flow_px = channel_flow_px_512 * s
        bank_outward_px = warp_step_px_512 * s

        road_color_mask = (
            (R > 0.62) & (G > 0.62) & (B > 0.56) &
            ((torch.max(base_t, dim=-1).values - torch.min(base_t, dim=-1).values) < 0.16)
        )
        road_band_mask = (signed_side > canal_half_width * 0.60) & (signed_side < canal_half_width + 22.0 * s)

        dense_tree_mask = (
            (G > R + 0.08) &
            (G > B + 0.05) &
            (G > 0.38) &
            (P[..., 0] > 0.60 * N)
        )

        road_mask = road_color_mask | road_band_mask
        static_block = road_mask | (dense_tree_mask & (dist_to_center > 24.0 * s))

        # 河道区域分成 core 与 bank
        source_mask = dist_to_center <= canal_half_width
        core_mask = dist_to_center <= canal_core_half_width
        bank_mask = (dist_to_center > canal_half_width) & (dist_to_center <= canal_half_width + 10.0 * s)

        # 道路侧扩张慢，农田侧正常
        side_speed = torch.where(signed_side > 0, road_slowdown, 1.0)
        effective_dist = (dist_to_center - canal_half_width) / torch.clamp(side_speed, min=1e-4)

        outward = perp / torch.clamp(dist_to_center.unsqueeze(-1), min=1e-4)
        downstream = d_unit.view(1, 1, 2).expand_as(outward)

        self.precomputed_flows = []
        self.flood_seq = []
        self.eta_seq = []

        identity_grid = (self.XY.to(torch.float32) / (self.N - 1)) * 2 - 1
        self.precomputed_flows.append(identity_grid)
        self.flood_seq.append(core_mask.float().unsqueeze(0).unsqueeze(0))
        self.eta_seq.append((0.85 * core_mask.float()).unsqueeze(0).unsqueeze(0))

        prev_mask = core_mask.clone()

        for i in range(num_frames):
            alpha = float(i + 1) / float(num_frames)
            radius = max_spread * (alpha ** 1.22)

            spread_mask = (
                (effective_dist > 0.0)
                & (effective_dist <= radius)
                & (~static_block)
            )
            curr_mask = core_mask | spread_mask
            front_mask = curr_mask & (~prev_mask)

            soft_core = self._blur_mask(core_mask.float(), k=7, passes=2)
            soft_bank = self._blur_mask(bank_mask.float(), k=7, passes=2)
            soft_curr = self._blur_mask(curr_mask.float(), k=7, passes=2)
            soft_front = self._blur_mask(front_mask.float(), k=9, passes=2)

            eta = torch.clamp(
                0.90 * soft_core +
                0.48 * soft_bank +
                0.30 * soft_curr +
                0.95 * soft_front,
                0.0,
                1.0,
            )

            channel_gate = torch.clamp(soft_core + 0.55 * soft_bank, 0.0, 1.0)
            overflow_gate = soft_curr * (0.50 + 0.50 * soft_front)

            disp = (
                downstream * channel_flow_px * channel_gate.unsqueeze(-1) +
                outward * bank_outward_px * overflow_gate.unsqueeze(-1) +
                downstream * (0.22 * bank_outward_px) * soft_front.unsqueeze(-1)
            ) / float(self.N)

            end_point = self.XY.to(torch.float32) - disp * float(self.N)
            end_point = end_point / float(self.N - 1)
            end_point = 2.0 * end_point - 1.0

            self.precomputed_flows.append(end_point)
            self.flood_seq.append(curr_mask.float().unsqueeze(0).unsqueeze(0))
            self.eta_seq.append(eta.unsqueeze(0).unsqueeze(0))
            prev_mask = curr_mask

    def _blur_mask(self, x: torch.Tensor, k: int = 7, passes: int = 1) -> torch.Tensor:
        x = x.unsqueeze(0).unsqueeze(0)
        pad = k // 2
        for _ in range(passes):
            x = F.avg_pool2d(x, kernel_size=k, stride=1, padding=pad)
        return x[0, 0]

    def get_default_image(self):
        return Image.open(self.image_path).convert("RGB")

    def get_default_framesteps(self) -> torch.Tensor:
        return torch.tensor(list(range(len(self.precomputed_flows))))

    def get_spatial_eta(self, t):
        return self.eta_seq[t]

    def get_flow(self, t) -> torch.Tensor:
        return self.precomputed_flows[t]